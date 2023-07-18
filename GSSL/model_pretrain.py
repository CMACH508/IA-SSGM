import torch
import torch.nn as nn
import GCL.augmentors as A
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, GINEConv, GATConv

from GSSL.gcl_utils import Augmentor, Encoder, normalize_over_channels, concat_features
from utils.feature_align import feature_align
from utils.config import cfg
import utils.backbone
CNN = eval('utils.backbone.{}'.format(cfg.BACKBONE))

class GCL(nn.Module):
    ''' implementation of GCL framework.
        encoder: GCL framework 中的 GNN_based Encoder. 
        
    '''
    def __init__(self):
        super(GCL, self).__init__()
        self.augmentor = Augmentor()
        # self.encoder = Encoder(input_dim = cfg.BIIA.FEATURE_CHANNEL, hidden_dim = cfg.BIIA.FEATURE_CHANNEL, num_layers = cfg.BIIA.GNN_LAYER)
        self.encoder = Encoder(input_dim=1024, hidden_dim=1024, num_layers=2)
        project_dim = cfg.BIIA.FEATURE_CHANNEL * 2 # TODO: 了解GINConv 看看输出为什么是这个维度
        # project head: 两层MLP
        self.project = nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(), # 对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量
            nn.Linear(project_dim, cfg.BIIA.FEATURE_CHANNEL)
        )
        self.gcl_params = list(self.parameters())
        self.rescale = cfg.PROBLEM.RESCALE

    def forward(self, x, edge_index, edge_attr, batch):
        # data augmentation
        x1, edge_index1, edge_attrs1, x2, edge_index2, edge_attrs2 = self.augmentor(x, edge_index, edge_attr)

        # encoder
        z, g = self.encoder(x, edge_index, edge_attr, batch = batch)
        z1, g1 = self.encoder(x1, edge_index1, edge_attrs1, batch = batch) #TODO: 测试graph有没有batch 属性
        z2, g2 = self.encoder(x2, edge_index2, edge_attrs2, batch = batch)

        # project head
        g1, g2 = [self.project(g) for g in [g1, g2]]
        z1, z2 = [self.project(z) for z in [z1, z2]]
        

        return z, g, z1, z2, g1, g2

    def forward_ori(self, data_dict):
        if 'images' in data_dict:
            images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']

        augmented_global_features = []
        augmented_local_features = []
        orig_graph_list = []
        
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # vgg16 pretrain model 作为初始化的 node representation (和 edge representation)
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features 
            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)

            node_features = torch.cat((U, F), dim=1)
            graph.x = node_features

            # data augmentation
            x1, edge_index1, edge_attrs1, x2, edge_index2, edge_attrs2 = self.augmentor(graph.x, graph.edge_index, graph.edge_attr)
    
            # encoder
            z, g = self.encoder(graph.x, graph.edge_index, graph.edge_attr,batch = graph.batch)
            z1, g1 = self.encoder(x1, edge_index1, graph.edge_attr, batch = graph.batch) #TODO: 测试graph有没有batch 属性
            z2, g2 = self.encoder(x2, edge_index2, graph.edge_attr,batch = graph.batch)
            
            
            # project head
            g1, g2 = [self.project(g) for g in [g1, g2]]
            z1, z2 = [self.project(z) for z in [z1, z2]]

            # 更新 graph.x, 即为更新graph的节点embedding.
            graph.x = z
           
            augmented_global_features.append([g1, g2])
            augmented_local_features.append([z1, z2]) 

            # 大图Batch变回成3张小图
            ori_graphs = graph.to_data_list()
            # 方便后续 matching 取用
            orig_graph_list.append(ori_graphs)

        # 重新拼接一下augmented_global_features，
        # [[g1_from_graph1, g2_from_graph1], [g1_from_graph2, g2_from_graph2]] --> [[g1_from_graph1, g2_from_graph1],
        #                                                                           [g1_from_graph2, g2_from_graph2],
        # 
        if cfg.TRAIN.LOSS_FUNC_MODE == 'G2G':
            g1_from_graph1, g2_from_graph1 = augmented_global_features[0]
            g1_from_graph2, g2_from_graph2 = augmented_global_features[1]
            concate_g1 = torch.cat((g1_from_graph1, g1_from_graph2),dim=0)
            concate_g2 = torch.cat((g2_from_graph1, g2_from_graph2))

            data_dict.update({
            'augmented_global_features': [concate_g1, concate_g2],
            'orig_graph_list': orig_graph_list,
        })

            
        elif cfg.TRAIN.LOSS_FUNC_MODE == 'L2L':
            z1_from_graph1, z2_from_graph1 = augmented_local_features[0]
            z1_from_graph2, z2_from_graph2 = augmented_local_features[1]
            concate_z1 = torch.cat((z1_from_graph1, z1_from_graph2),dim=0)
            concate_z2 = torch.cat((z2_from_graph1, z2_from_graph2))

            data_dict.update({
                'augmented_local_features': [concate_z1, concate_z2],
                'orig_graph_list': orig_graph_list,
            })
        else:
            raise ValueError('Unknown loss function mode {}'.format(cfg.TRAIN.LOSS_FUNC_MODE))


        return data_dict