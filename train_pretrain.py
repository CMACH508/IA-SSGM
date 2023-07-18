import os.path as osp
from pathlib import Path
from datetime import datetime
import time
from tensorboardX import SummaryWriter
import GCL.losses as L
import torch
import torch.optim as optim
import torch.utils.data as Data
from torch_geometric.loader import DataLoader

from parallel import DataParallel
from data.Pascal_voc_ssl import PascalVOCKeypoints
from data.willow_obj_ssl import WILLOWObjectClass
from data.imc_pt_sparsegm_ssl import IMCPTSparseGMObject
from data.cub2011_ssl import CUB2011Object
from data.full_connected import FullConnected
from data.hippi_distance import HIPPIDistance
from GCL.models.contrast_model import DualBranchContrast

from utils.decoupled_infonce import DecoupledInfoNCE, INfoNCE
from utils.dup_stdout_manager import DupStdoutFileManager
from utils.print_easydict import print_easydict
from utils.model_sl import load_model, save_model
from utils.parse_args import parse_args
from utils.config import cfg

def train_eval_model(model,
                     criterion, 
                     optimizer,
                     train_loader,
                     test_loader,
                     tfboard_writter, 
                     num_epochs=25,
                     start_epoch = 0):
    print("Start training GSSL . . .")
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
            with torch.set_grad_enabled(True):
                z, g, z1, z2, g1, g2 = model(data.x, data.edge_index, data.edge_attr, data.batch)
                # loss = criterion(h1 = z1, h2 = z2)
                loss = criterion(h1 = z1, h2 = z2, g1 = g1, g2 = g2, batch = data.batch)
                loss.backward()
                optimizer.step()

                loss_dict = dict()
                loss_dict['loss'] = loss.item()
                tfboard_writter.add_scalars('loss', loss_dict, epoch * (len(train_loader.dataset) / (data.batch.max().item() + 1))+ iter_num)

                running_loss += loss.item() * (data.batch.max().item() + 1)
                epoch_loss += loss.item() * (data.batch.max().item() + 1)

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * (data.batch.max().item() + 1) / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'.format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / (data.batch.max().item() + 1)))
                    running_loss = 0.0
                    running_since = time.time()
        
        epoch_loss = epoch_loss / len(train_loader.dataset)
        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))
        
        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()

        scheduler.step()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    args = parse_args('GSSL PreTrain')
    
    import importlib
    mod = importlib.import_module(cfg.MODULE) # 一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行
    torch.manual_seed(cfg.RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mod.GCL().to(device)

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
    # # pascalvoc
    # for category in datasets.categories:
    #     dataset = datasets(cfg.DATASET_PATH, category, train=True, pre_transform=pre_transform, pre_filter=pre_filter)
    #     train_datasets.append(dataset)
    #     dataset = datasets(cfg.DATASET_PATH, category, train=False, pre_transform=pre_transform, pre_filter=pre_filter)
    #     test_datasets.append(dataset)
    # train_dataset = Data.ConcatDataset(train_datasets)
    # test_dataset = Data.ConcatDataset(test_datasets)
    # train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    # # willow
    # datasets = [datasets(cfg.DATASET_PATH, cat, pre_transform) for cat in datasets.categories]
    # datasets = [dataset.shuffle() for dataset in datasets]
    # train_datasets = [dataset[:-20] for dataset in datasets]
    # test_datasets = [dataset[-20:] for dataset in datasets]
    # train_dataset = Data.ConcatDataset(train_datasets)
    # train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, follow_batch=['x_s', 'x_t'])
    # test_dataset = Data.ConcatDataset(test_datasets)
    # test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    # imcpt
    train_datasets = [datasets(cfg.DATASET_PATH, cat, pre_transform=pre_transform) for cat in datasets.categories[:-3]]
    train_datasets = [dataset.shuffle() for dataset in train_datasets]
    test_datasets = [datasets(cfg.DATASET_PATH,cat, pre_transform=pre_transform) for cat in datasets.categories[-3:]]
    test_datasets = [dataset.shuffle() for dataset in test_datasets]
    train_dataset =  Data.ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_dataset = Data.ConcatDataset(test_datasets)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    # # cub2011
    # for category in datasets.categories:
    #     dataset = datasets(cfg.DATASET_PATH, category, train=True, pre_transform=pre_transform, pre_filter=pre_filter)
    #     train_datasets.append(dataset)
    #     dataset = datasets(cfg.DATASET_PATH, category, train=False, pre_transform=pre_transform, pre_filter=pre_filter)
    #     test_datasets.append(dataset)
    # train_dataset = Data.ConcatDataset(train_datasets)
    # test_dataset = Data.ConcatDataset(test_datasets)
    # train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    criterion = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode=cfg.TRAIN.LOSS_FUNC_MODE)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(model, criterion, optimizer, train_loader, test_loader, tfboardwriter, start_epoch=cfg.TRAIN.START_EPOCH, num_epochs=cfg.TRAIN.NUM_EPOCHS)