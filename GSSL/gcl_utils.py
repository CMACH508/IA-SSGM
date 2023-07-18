# 这里用Graph Contrastive learning的方式，来提取 graph 特征
# vgg16 relu 4_2 和 relu 5_1 作为input graph 的node representation 和edge representation。
# @inproceedings{You2020GraphCL,
#  author = {You, Yuning and Chen, Tianlong and Sui, Yongduo and Chen, Ting and Wang, Zhangyang and Shen, Yang},
#  booktitle = {Advances in Neural Information Processing Systems},
#  editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
#  pages = {5812--5823},
#  publisher = {Curran Associates, Inc.},
#  title = {Graph Contrastive Learning with Augmentations},
#  url = {https://proceedings.neurips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf},
#  volume = {33},
#  year = {2020}
# }
# 
## 之前的data_loader.py 传入网络的是[image1, image2], 处理的是一对image.
## 而GCL处理的是单个的image，所以 forward 里面单独处理。

from turtle import forward
import torch
import torch.nn as nn
import GCL.augmentors as A
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, GINEConv, GATConv, SplineConv

from utils.feature_align import feature_align
from utils.config import cfg


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
    # forward(x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_sparse.tensor.SparseTensor], size: Optional[Tuple[int, int]] = None) → torch.Tensor[source]

def make_gine_conv(input_dim, out_dim, edge_dim=None):
    return GINEConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)), train_eps=True, edge_dim=edge_dim)

def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms

def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip (embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

class Augmentor(nn.Module):
    def __init__(self):
        super(Augmentor, self).__init__()
        # # cub
        # self.augmentor1 = A.Compose([A.RWSampling(num_seeds=12, walk_length=4), A.FeatureMasking(pf=0.3)])
        # self.augmentor2 = A.Compose([A.RWSampling(num_seeds=12, walk_length=4), A.FeatureMasking(pf=0.3)])
        # imcpt
        self.augmentor1 = A.Compose([A.NodeDropping(pn=0.5), A.EdgeRemoving(pe=0.5)]) 
        self.augmentor2 = A.Compose([A.NodeDropping(pn=0.5), A.EdgeRemoving(pe=0.5)])
        # # cmu 
        # self.augmentor1 = A.Compose([A.EdgeRemoving(pe=0.3), A.EdgeAttrMasking(pf=0.3)])
        # self.augmentor2 = A.Compose([A.EdgeRemoving(pe=0.3), A.EdgeAttrMasking(pf=0.3)])
        # # pascalVOC /Willow
        # self.augmentor1 = A.Compose([A.EdgeRemoving(pe=0.3), A.EdgeAttrMasking(pf=0.5), A.FeatureMasking(pf=0.5),])
        # self.augmentor2 = A.Compose([A.EdgeRemoving(pe=0.3), A.EdgeAttrMasking(pf=0.5), A.FeatureMasking(pf=0.5),])

    def forward(self, x, edge_index, edge_attr):
        x1, edge_index1, edge_attrs1 = self.augmentor1(x, edge_index, edge_attr)
        x2, edge_index2, edge_attrs2 = self.augmentor2(x, edge_index, edge_attr)
        return x1, edge_index1, edge_attrs1, x2, edge_index2, edge_attrs2

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers =  num_layers
        for i in range(num_layers):
            if i == 0:
                if cfg.GCL.ENCODER_TYPE == 'GINEConv': 
                    self.layers.append(make_gine_conv(input_dim, hidden_dim, edge_dim=2))
                elif cfg.GCL.ENCODER_TYPE == 'GATConv':
                    self.layers.append(GATConv(in_channels = input_dim, out_channels = hidden_dim, heads = 4, concat = False, edge_dim = 2))
                elif cfg.GCL.ENCODER_TYPE == 'SplineCNN':
                    self.layers.append(SplineConv(in_channels = input_dim, out_channels = hidden_dim, dim = 2, kernel_size=5)) 
                elif cfg.GCL.ENCODER_TYPE == 'Spline+GINEConv':
                    self.layers.append(SplineConv(in_channels = input_dim, out_channels = hidden_dim, dim = 2, kernel_size=5))
            else:
                if cfg.GCL.ENCODER_TYPE == 'GINEConv':
                    self.layers.append(make_gine_conv(hidden_dim, hidden_dim, edge_dim=2))
                elif cfg.GCL.ENCODER_TYPE == 'GATConv':
                    self.layers.append(GATConv(in_channels = hidden_dim, out_channels = hidden_dim, heads = 4, concat = False, edge_dim = 2))
                elif cfg.GCL.ENCODER_TYPE == 'SplineCNN':
                    self.layers.append(SplineConv(in_channels= hidden_dim, out_channels=hidden_dim, dim= 2, kernel_size=5))
                elif cfg.GCL.ENCODER_TYPE == 'Spline+GINEConv':
                    self.layers.append(make_gine_conv(hidden_dim, hidden_dim, edge_dim=2))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.encoder_params = list(self.parameters())
    # # encoder == 'splinecnn'
    # def forward(self, x, edge_index, edge_attr, batch):
    #     z = x
    #     for i, (conv, bn) in enumerate(zip(self.layers, self.batch_norms)):
    #         # z = bn(z)
    #         z = conv(z, edge_index, edge_attr)
    #         if i < self.num_layers - 1:
    #             z = F.elu(z)
    #         if i == self.num_layers - 1:
    #             z = F.dropout(z, p=0.6)  

    #     g = global_add_pool(z, batch)

    #     return z, g
    
    # encoder == 'GINEconv'
    def forward(self, x, edge_index, edge_attr, batch):
        z = x
        zs = []
        
        for i, (conv, bn) in enumerate(zip(self.layers, self.batch_norms)):
            z = conv(z, edge_index, edge_attr)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)

        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g
        

            

            


        

