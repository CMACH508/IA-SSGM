import os
import os.path as osp
from pathlib import Path
from datetime import datetime
import time
from numpy.random.mtrand import random, sample
import xlwt
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch_geometric.loader import DataLoader
from yaml import parse

from parallel import DataParallel
from data.Pascal_voc_ssl import PascalVOCKeypoints
from data.willow_obj_ssl import WILLOWObjectClass
from data.imc_pt_sparsegm_ssl import IMCPTSparseGMObject
from data.cub2011_ssl import CUB2011Object
from data.data_loader_pairdata import SamePairDataset, ValidPairDataset, PairDataset
from data.full_connected import FullConnected
from data.hippi_distance import HIPPIDistance
from utils.permutation_loss import CrossEntropyLoss
from utils.dup_stdout_manager import DupStdoutFileManager
from utils.generate_gt import generate_y
from utils.print_easydict import print_easydict
from utils.evaluation_metric import matching_accuracy
from utils.model_sl import load_model, save_model
from eval_finetune import eval_model
from utils.parse_args import parse_args
from utils.config import cfg

def train_eval_model(model, 
                     criterion, 
                     optimizer, 
                     train_loader, 
                     test_datasets, 
                     tfboardwriter, 
                     start_epoch, 
                     num_epochs, 
                     xls_wb):
    print("Start training Matching . . .")
    since = time.time()

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True) # parents= True, 创建这个路径的任何缺失的父目录

    #  断点恢复
    model_path, optim_path = '', ''
    if start_epoch > 0:
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
    
    if len(model_path) > 0:
        print('Loadding model from {}'.format(model_path))
        load_model(model, model_path, strict=False)
    if len(optim_path) > 0:
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))
    
    # lr_decay.
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY,
                                               last_epoch=cfg.TRAIN.START_EPOCH - 1) # 按照设定的间隔调整学习率,其中gamma为倍数
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print ('-'*10)

        model.train() # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0 # 每输出一次/每statistic_step次 iter_num 的loss 
        running_since = time.time()
        iter_num = 0
        for data in train_loader:
            iter_num = iter_num + 1
            optimizer.zero_grad()
            data = data.to(device)
            with torch.autograd.set_detect_anomaly(True):
                pred_softmax_m, pred_permutation_m, s_num_node, t_num_node, probability, cross_probability = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch, data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
                # pred_softmax_m, pred_permutation_m, pseudo_ground_truth_m, s_num_node, t_num_node = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch, data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch, device)
                # ground_truth_m = torch.stack([torch.eye(int(s_num_node[i])) for i in range(len(s_num_node))], dim=0)
                ground_truth_m = generate_y(data.y, data.x_s_batch, s_num_node, t_num_node).to(device)

                
                # loss = criterion(pred_softmax_m, pred_permutation_m, s_num_node, t_num_node)
                loss1 = criterion(pred_softmax_m, pred_permutation_m, s_num_node, t_num_node).to(device)
                loss2 = F.cross_entropy(probability, cross_probability, reduction='mean')
                loss = loss1 + loss2
                # loss = criterion(pred_softmax_m, pseudo_ground_truth_m, s_num_node, t_num_node)
                loss.backward()
                optimizer.step() 

                # loss 
                loss_dict = dict()
                loss_dict['loss'] = loss.item()
                loss_dict['loss_matching'] = loss1.item()
                loss_dict['loss_consistency'] = loss2.item()
                tfboardwriter.add_scalars('loss', loss_dict, epoch * (len(train_loader.dataset) / (data.x_s_batch.max().item() + 1))+ iter_num)
                # acc
                
                acc, _, _ = matching_accuracy(pred_permutation_m, ground_truth_m, s_num_node)

                acc_dict = dict()
                acc_dict['matching_accuracy'] = torch.mean(acc)
                tfboardwriter.add_scalars(
                    'training_acc',
                    acc_dict,
                    epoch * (len(train_loader.dataset) / (data.x_s_batch.max().item() + 1))+ iter_num
                )

                running_loss += loss.item() * (data.x_s_batch.max().item() + 1)
                epoch_loss += loss.item() * (data.x_s_batch.max().item() + 1)

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * (data.x_s_batch.max().item() + 1) / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'.format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / (data.x_s_batch.max().item() + 1 )))
                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / len(train_loader.dataset)
        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))
        
        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()

        # Eval in each epoch
        datasets = eval(cfg.DATASET_FULL_NAME)
        recalls = eval_model(model, test_datasets, device, xls_sheet=xls_wb.add_sheet('epoch{}'.format(epoch + 1)))
        recall_dict = {"{}".format(category): single_recall for category, single_recall in zip(datasets.categories, recalls)}
        recall_dict['average'] = torch.mean(recalls)
        tfboardwriter.add_scalars(
            'evaluating_acc',
            recall_dict,
            epoch * (len(train_loader.dataset) / (data.x_s_batch.max().item() + 1))
        ) 
        wb.save(wb.__save_path)

        scheduler.step()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model

if __name__ == '__main__':
    args = parse_args('GSSL finetune')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import importlib
    mod = importlib.import_module(cfg.MODULE)
    torch.manual_seed(cfg.RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mod.Matching_cross_attention().to(device)
    import torch_geometric.transforms as T
    pre_filter = lambda data: data.pos.size(0) > 1
    # pre_transform = T.Compose([
    #     T.Delaunay(),   # Computes the delaunay triangulation of a set of points.
    #     T.FaceToEdge(), # Converts mesh faces [3, num_faces] to edge indices [2, num_edges].
    #     T.Cartesian(),  # Saves the relative Cartesian coordinates of linked nodes in its edge attributes.
    # ])
    # pre_transform = T.Compose([
    #     T.Delaunay(),
    #     T.FaceToEdge(),
    #     HIPPIDistance(),
    # ])
    
    pre_transform = T.Compose([
        FullConnected(),
        T.Cartesian(),
    ])
    
    train_datasets = []
    test_datasets = []

    datasets = eval(cfg.DATASET_FULL_NAME)
    # # pascalVoc
    # for category in datasets.categories:
    #     dataset = datasets(cfg.DATASET_PATH, category, train=True, pre_transform=pre_transform, pre_filter=pre_filter)
    #     train_datasets += [SamePairDataset(dataset, dataset, train=True, random=False)]
    #     dataset = datasets(cfg.DATASET_PATH, category, train=False, pre_transform=pre_transform, pre_filter=pre_filter)
    #     test_datasets += [SamePairDataset(dataset, dataset, train=False, random=False)]
    # train_dataset = Data.ConcatDataset(train_datasets)
    # train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, follow_batch=['x_s', 'x_t'])
    
    # # willow
    # datasets = [datasets(cfg.DATASET_PATH, cat, pre_transform) for cat in datasets.categories]
    # datasets = [dataset.shuffle() for dataset in datasets]
    # train_datasets = [dataset[:-20] for dataset in datasets]
    # test_datasets = [dataset[-20:] for dataset in datasets]
    # train_datasets = [
    #     PairDataset(train_dataset, train_dataset, sample=False)
    #     for train_dataset in train_datasets
    # ]   
    # train_dataset = Data.ConcatDataset(train_datasets)
    # train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, follow_batch=['x_s', 'x_t'])
    # test_datasets = [
    #     PairDataset(test_dataset, test_dataset) for test_dataset in test_datasets
    # ]

    # # willow 2
    # datasets = [datasets(cfg.DATASET_PATH, cat, pre_transform) for cat in datasets.categories]
    # datasets = [dataset.shuffle() for dataset in datasets]
    # train_datasets = [dataset[:20] for dataset in datasets]
    # test_datasets = [dataset[20:] for dataset in datasets]
    # train_datasets = [
    #     SamePairDataset(train_dataset, train_dataset, train=True, random=False)
    #     for train_dataset in train_datasets
    # ]
    # train_dataset = Data.ConcatDataset(train_datasets)
    # train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, follow_batch=['x_s', 'x_t'])
    # test_datasets =[
    #    SamePairDataset(test_dataset, test_dataset, train=False, random=False)
    #     for test_dataset in test_datasets
    # ]


    # # imcpt
    # for category in datasets.categories[:-3]:
    #     dataset = datasets(cfg.DATASET_PATH, category, pre_transform=pre_transform)
    #     train_datasets += [SamePairDataset(dataset, dataset, train=True, random=False)]

    # for category in datasets.categories[-3:]:
    #     dataset = datasets(cfg.DATASET_PATH, category, pre_transform=pre_transform)
    #     test_datasets += [SamePairDataset(dataset, dataset, train=False, random=False)]
    # train_dataset = Data.ConcatDataset(train_datasets)
    # train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, follow_batch=['x_s', 'x_t'])
    
    # cub2011
    for category in datasets.categories:
        dataset = datasets(cfg.DATASET_PATH, category, train=True, pre_transform=pre_transform, pre_filter=pre_filter)
        train_datasets += [SamePairDataset(dataset, dataset, train=True, random=False)]
        dataset = datasets(cfg.DATASET_PATH, category, train=False, pre_transform=pre_transform, pre_filter=pre_filter)
        test_datasets += [SamePairDataset(dataset, dataset, train=False, random=False)]
    train_dataset = Data.ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, follow_batch=['x_s', 'x_t'])


    criterion = CrossEntropyLoss()
    
    if cfg.TRAIN.SEPARATE_BACKBONE_LR:
        encoder_ids = [id(item) for item in model.encoder.encoder.encoder_params]
        other_params = [param for param in model.parameters() if id(param) not in encoder_ids]
        model_params = [
            {'params': other_params},
            {'params': model.encoder.encoder.encoder_params, 'lr': cfg.TRAIN.BACKBONE_LR}
        ]
    else:
        model_params = model.parameters()
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))

    wb = xlwt.Workbook()  # 新建一个excel 文件
    wb.__save_path = str(Path(cfg.OUTPUT_PATH) / ('train_eval_result_' + now_time + '.xls'))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(model, criterion, optimizer, train_loader, test_datasets, tfboardwriter, start_epoch=cfg.TRAIN.START_EPOCH, num_epochs=cfg.TRAIN.NUM_EPOCHS, xls_wb=wb)
    
    
    
