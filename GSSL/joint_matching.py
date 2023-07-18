import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GatedGraphConv

from GSSL.model_pretrain import GCL
from GSSL.probability_layer import FullPro
from utils.pad_tensor import pad_tensor
from utils.model_sl import load_model
from utils.sinkhorn import Sinkhorn
from utils.hungarian import hungarian
from utils.config import cfg

def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out

def to_sparse(x, mask):
    return x[mask]


class joint_matching(nn.Module):
    '''
    joint graph contrastive learning framework with 
    graph matching pipeline.
    '''
    def __init__(self):
        super(joint_matching, self).__init__()
        self.gcl = GCL()
        self.voting_layer = FullPro()
        self.hungarian = hungarian
        self.__iteration = cfg.BIIA.ITERATION_
        
    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s, 
                      x_t, edge_index_t, edge_attr_t, batch_t):
        z_s, g_s, z1_s, z2_s, g1_s, g2_s = self.gcl(x_s, edge_index_s, edge_attr_s, batch_s)
        z_t, g_t, z1_t, z2_t, g1_t, g2_t = self.gcl(x_t, edge_index_t, edge_attr_t, batch_t)
        
        z_s, s_mask = to_dense_batch(z_s, batch_s, fill_value=0)
        z_t, t_mask = to_dense_batch(z_t, batch_t, fill_value=0)

        s_num_node = s_mask.sum(dim=-1)
        t_num_node = t_mask.sum(dim=-1)
        (batch_size, N_s, _), N_t = z_s.size(), z_t.size(1)

        assert z_s.size(0) == z_t.size(0), 'Encountered unequal batch-sizes'

        unary_affs = torch.matmul(z_s, z_t.transpose(-2, -1))
        probability = self.voting_layer(unary_affs, s_num_node, t_num_node)

        cross_binary_m = binary_m = self.hungarian(probability, s_num_node, t_num_node)
        
        # last_cross_z_s = z_s # B X N_s X Feature
        # last_cross_z_t = z_t
        last_cross_z_s = to_dense(x_s, s_mask)
        last_cross_z_t = to_dense(x_t, t_mask)
        

        for k in range(self.__iteration):
            cross_z_s = to_sparse(torch.bmm(cross_binary_m, last_cross_z_t), s_mask) # (N1+N2+...) x Feature
            cross_z_t = to_sparse(torch.bmm(cross_binary_m.transpose(1, 2), last_cross_z_s), t_mask) 
            
            cross_z_s, _, _, _, _, _ = self.gcl(cross_z_s, edge_index_s, edge_attr_s, batch_s)
            cross_z_t, _, _, _, _, _ = self.gcl(cross_z_t, edge_index_t, edge_attr_t, batch_t)
            
            cross_z_s, cross_z_t = to_dense(cross_z_s, s_mask), to_dense(cross_z_t, t_mask)

            assert cross_z_s.size(0) == cross_z_s.size(0), 'Encountered unequal batch-sizes'

            cross_unary_affs = torch.matmul(cross_z_s, cross_z_t.transpose(-2, -1))
            cross_probability = self.voting_layer(cross_unary_affs, s_num_node, t_num_node)
            
        return z1_s, z2_s, g1_s, g2_s, z1_t, z2_t, g1_t, g2_t, probability, binary_m, s_num_node, t_num_node, probability, cross_probability
            
        