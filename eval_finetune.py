import time
import torch
import random
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.utils import to_dense_batch

from data.Pascal_voc_ssl import PascalVOCKeypoints
from data.willow_obj_ssl import WILLOWObjectClass
from data.imc_pt_sparsegm_ssl import IMCPTSparseGMObject
from data.cub2011_ssl import CUB2011Object
from data.cmu_ssl import CMUHouseHotel

from utils.generate_gt import generate_y
from utils.evaluation_metric import *
from utils.visual_result import show_results
from utils.visible_graph import visible_compare_pair_graph
from utils.config import cfg

def eval_model(model, 
               test_datasets,
               device,
               xls_sheet=None,
               visible=False):
    print('Start Eval finetune . . . ')
    since = time.time()
    was_training = model.training
    model.eval()

    recalls, precisions, f1s = [], [], []
    for test_dataset in test_datasets:
        test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, follow_batch=['x_s', 'x_t'])
        # test_loader1 = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=True)
        # test_loader2 = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=True)
        recall_list, precision_list, f1_list = [], [], []
        # for data_s, data_t in zip(test_loader1, test_loader2):
        #     data_s, data_t = data_s.to(device), data_t.to(device)
        for data in test_loader:
            data = data.to(device)
            with torch.set_grad_enabled(False):
                # pred_softmax_m, pred_permutation_m, pseudo_ground_truth_m, s_num_node, t_num_node = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch, data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch, device)
                # pred_softmax_m, pred_permutation_m, s_num_node, t_num_node, _, _ = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch, data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
                _, _, _, _, _, _, _, _, pred_softmax_m, pred_permutation_m, s_num_node, t_num_node, _, _ = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch, data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
                # pred_softmax_m, pred_permutation_m, s_num_node, t_num_node = model(data_s.x, data_s.edge_index, data_s.edge_attr,
                #                                                                     data_s.batch, data_t.x, data_t.edge_index,
                #                                                                     data_t.edge_attr, data_t.batch)
            
            # acc
            ground_truth_m = generate_y(data.y, data.x_s_batch, s_num_node, t_num_node)
            # ground_truth_m = torch.stack([torch.eye(int(s_num_node[i])) for i in range(len(s_num_node))], dim=0)
            recall, _, _ = matching_accuracy(pred_permutation_m, ground_truth_m, s_num_node)
            recall_list.append(recall)
            
            precision, _, __ = matching_precision(pred_permutation_m, ground_truth_m, s_num_node)
            precision_list.append(precision)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1[torch.isnan(f1)] = 0
            f1_list.append(f1)

            if visible and torch.mean(recall) > 0.6:
                print('visible...') 
                visible_compare_pair_graph([data.img_s, data.img_t],   
                                           [data.pos_s, data.pos_t],
                                           [data.y_s, data.y_t],
                                           [data.edge_index_s, data.edge_index_t],
                                           [data.x_s_batch, data.x_t_batch],
                                           device,
                                           [data.name_s, data.name_t],
                                           pred_permutation_m,
                                           ground_truth_m)
                

            # if visible and random.random() < 0.3:
            #     print('visible . . . ')
            #     print(data.img_s[0])
            #     print(data.img_t[0])
            #     visible_compare_pair_graph([data.img_s, data.img_t],
            #                                 [data.pos_s, data.pos_t],
            #                                 [data.y_s, data.y_t],
            #                                 [data.edge_index_s, data.edge_index_t],
            #                                 [data.x_s_batch, data.x_t_batch],
            #                                 device,
            #                                 [data.name_s, data.name_t],
            #                                 pred_permutation_m,
            #                                 ground_truth_m)
            #     # std = torch.tensor([0.229, 0.224, 0.225], device=device)
            #     # mean = torch.tensor([0.485, 0.456, 0.406], device=device)
            #     # img_src = data.img_s.view(-1, 3, 256, 256)
            #     # img_src = img_src.permute(0, 2, 3, 1) * std + mean
                
            #     # img_tgt = data.img_t.view(-1, 3, 256, 256)
            #     # img_tgt = img_tgt.permute(0, 2, 3, 1) * std + mean
                
                
            #     # pos_src, mask = to_dense_batch(data.pos_s, batch=data.x_s_batch, fill_value=0)
            #     # pos_tgt, mask = to_dense_batch(data.pos_t, batch=data.x_t_batch ,fill_value=0)
            #     # show_results([img_src.permute(0, 3, 1, 2), img_tgt.permute(0, 3, 1, 2)], [pos_src, pos_tgt], pred_permutation_m, ground_truth_m.cuda(), recall='low_recall', Fns=[data.name_s, data.name_t])
                
        if len(recall_list) == 0:
            print("not evaluate {}".format(test_dataset.dataset_s.category))

        recalls.append(torch.cat(recall_list))
        precisions.append(torch.cat(precision_list))
        f1s.append(torch.cat(f1_list))


    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode = was_training)

    datasets = eval(cfg.DATASET_FULL_NAME)
    if xls_sheet:
        # 表头：列头
        for idx, category in enumerate(datasets.categories):
            xls_sheet.write(0, idx + 1, category)
        xls_sheet.write(0, idx + 2, 'mean')
        # 表头: 行头
        xls_row = 1
        xls_sheet.write(xls_row, 0, 'precision')
        xls_sheet.write(xls_row + 1, 0, 'recall')
        xls_sheet.write(xls_row + 2, 0, 'f1')       

    for idx, (category, category_pre, category_rec, category_f1) in enumerate(zip(datasets.categories, precisions, recalls, f1s)):
        print('{}: {}'.format(category, format_accuracy_metric(category_pre, category_rec, category_f1))) 
        if xls_sheet:
            xls_sheet.write(xls_row, idx + 1, torch.mean(category_pre).item())
            xls_sheet.write(xls_row + 1, idx + 1, torch.mean(category_rec).item())
            xls_sheet.write(xls_row + 2, idx + 1, torch.mean(category_f1).item())
    print('mean accuracy: {}'.format(format_accuracy_metric(torch.cat(precisions), torch.cat(recalls), torch.cat(f1s))))
    if xls_sheet:
        xls_sheet.write(xls_row, idx + 2, torch.mean(torch.cat(precisions)).item())
        xls_sheet.write(xls_row + 1, idx + 2, torch.mean(torch.cat(recalls)).item())
        xls_sheet.write(xls_row + 2, idx + 2, torch.mean(torch.cat(f1s)).item())
    
    return torch.Tensor(list(map(torch.mean, recalls)))        

if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict
    from utils.count_model_params import count_parameters

    args = parse_args('Test the GSSL Model')
    
    from GSSL.Matching import Matching
    from parallel import DataParallel
    import importlib
    mod = importlib.import_module(cfg.MODULE)
    # torch.manual_seed(cfg.RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mod.joint_matching().to(device)

    model = DataParallel(model, device_ids=cfg.GPUS)
    from data.Pascal_voc_ssl import PascalVOCKeypoints
    from data.willow_obj_ssl import WILLOWObjectClass
    from data.cub2011_ssl import CUB2011Object
    from data.data_loader_pairdata import ValidPairDataset, PairDataset, SamePairDataset
    from data.full_connected import FullConnected
    from data.hippi_distance import HIPPIDistance
    import torch_geometric.transforms as T
    from utils.model_sl import load_model
    
    pre_filter = lambda data: data.pos.size(0) > 1
    pre_transform = T.Compose([
        T.Delaunay(),   # Computes the delaunay triangulation of a set of points.
        T.FaceToEdge(), # Converts mesh faces [3, num_faces] to edge indices [2, num_edges].
        T.Cartesian(),  # Saves the relative Cartesian coordinates of linked nodes in its edge attributes.
    ])

    # pre_transform = T.Compose([
    #     FullConnected(),
    #     T.Cartesian(),
    # ])
    # pre_transform = T.Compose([
    #     T.Delaunay(),
    #     T.FaceToEdge(),
    #     HIPPIDistance(),
    # ])
    
    test_datasets = []
    # pascalvoc
    # for category in PascalVOCKeypoints.categories:
    category = 'horse'
    dataset = PascalVOCKeypoints(cfg.DATASET_PATH, category, train=False, pre_transform=pre_transform, pre_filter=pre_filter)
    test_datasets += [SamePairDataset(dataset, dataset, train=False, random=False)]

    # # willow
    # datasets = [WILLOWObjectClass(cfg.DATASET_PATH, cat, pre_transform) for cat in WILLOWObjectClass.categories]
    # datasets = [dataset.shuffle() for dataset in datasets]
    # test_datasets = [dataset[20:] for dataset in datasets]
    # test_datasets =[
    #    SamePairDataset(test_dataset, test_dataset, train=False, random=False)
    #     for test_dataset in test_datasets
    # ]

    # # cub2011-2
    # for category in CUB2011Object.categories:
    #     dataset = CUB2011Object(cfg.DATASET_PATH, category, train=False, pre_transform=pre_transform, pre_filter=pre_filter)
    #     test_datasets += [SamePairDataset(dataset, dataset, train=False, random=False)]
    
    # # cub2011
    # for category in CUB2011Object.categories:
    #     dataset = CUB2011Object(cfg.DATASET_PATH, category, train=False, pre_transform=pre_transform, pre_filter=pre_filter)
    #     test_datasets += [ValidPairDataset(dataset, dataset, sample=True, random=False)]
   
    # # cmu
    # datasets = [CMUHouseHotel(cfg.DATASET_PATH, cat, pre_transform) for cat in CMUHouseHotel.categories]
    # datasets = [dataset.shuffle() for dataset in datasets]
    # test_datasets = [dataset[70:] for dataset in datasets]
    # test_datasets = [
    #     SamePairDataset(test_dataset, test_dataset, train=False, random=False)
    #     for test_dataset in test_datasets
    # ]

    # #imcpt
    # for category in IMCPTSparseGMObject.categories[-3:]:
    #     dataset = IMCPTSparseGMObject(cfg.DATASET_PATH, category, pre_transform=pre_transform)
    #     test_datasets += [SamePairDataset(dataset, dataset, train=False, random=False)]
    # #imcpt
    # for category in IMCPTSparseGMObject.categories[-3:]:
    #     dataset = IMCPTSparseGMObject(cfg.DATASET_PATH, category, pre_transform=pre_transform)
    #     test_datasets += [ValidPairDataset(dataset, dataset, sample=False, random=True)]
    

    model_path = cfg.PRETRAINED_PATH
    load_model(model, model_path, strict=False)
    eval_model(model, test_datasets, device, visible=True)

    # datasets = [WILLOWObjectClass(cfg.DATASET_PATH, cat, pre_transform) for cat in WILLOWObjectClass.categories]
    # datasets = [dataset.shuffle() for dataset in datasets]
    # train_datasets = [dataset[:20] for dataset in datasets]
    # test_datasets = [dataset[20:] for dataset in datasets]
    # for i in range(len(datasets)):
    #     print(len(datasets[i]))
    # exit()
    # train_datasets = [
    #     PairDataset(train_dataset, train_dataset, sample=False)
    #     for train_dataset in train_datasets
    # ]   
    # # train_dataset = Data.ConcatDataset(train_datasets)
    # # train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, follow_batch=['x_s', 'x_t'])
    # test_datasets = [
    #     PairDataset(test_dataset, test_dataset, sample=True) for test_dataset in test_datasets
    # ]
    # print(test_datasets)
    # model_path = 'output/gssl_willow_finetune/params/params_0021.pt'
    # load_model(model, model_path, strict=False)
    # eval_model(model, test_datasets, device)