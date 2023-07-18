# 这部分用来处理，匹配问题，
# 调用pre_train 之后的model_pretrain.augmentor 和 model_pretrain.encoder,
# 得到新的 graph.x

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

def lexico_iter(lex):
    """返回由输入 iterable 中元素组成长度为 2 的子序列。
    """
    return itertools.combinations(lex, 2)

def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out

def to_sparse(x, mask):
    return x[mask]

class Matching(nn.Module):
    def __init__(self):
        super(Matching, self).__init__()
        self.encoder = GCL()
        if len(cfg.PRETRAINED_PATH) > 0:
            encoder_model_path = cfg.PRETRAINED_PATH
            print('Loading model from {}'.format(encoder_model_path))
            load_model(self.encoder, encoder_model_path, strict=False)
        if cfg.TRAIN.BACKBONE_LR == 0.0:
            for p in self.encoder.parameters():
                p.requires_grad = False
        
        
        self.affinity_layer = Affinity(cfg.BIIA.FEATURE_CHANNEL)
        self.voting_layer = FullPro()
        self.bi_stochastic = Sinkhorn()
        self.hungarian = hungarian

        self.alpha_1 = cfg.BIIA.ALPHA1
        self.alpha_2 = cfg.BIIA.ALPHA2
        self.__iteration = cfg.BIIA.ITERATION_
    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s, x_t, edge_index_t, edge_attr_t, batch_t):
        r"""
        Args:
            x_s (Tensor): Source graph node features of shape
                :obj:`[batch_size * num_nodes, C_in]`.
            edge_index_s (LongTensor): Source graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_s (Tensor): Source graph edge features of shape
                :obj:`[num_edges, D]`. Set to :obj:`None` if the GNNs are not
                taking edge features into account.
            batch_s (LongTensor): Source graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Set to :obj:`None` if operating on single graphs.
            x_t (Tensor): Target graph node features of shape
                :obj:`[batch_size * num_nodes, C_in]`.
            edge_index_t (LongTensor): Target graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_t (Tensor): Target graph edge features of shape
                :obj:`[num_edges, D]`. Set to :obj:`None` if the GNNs are not
                taking edge features into account.
            batch_t (LongTensor): Target graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Set to :obj:`None` if operating on single graphs.
        Returns:

        """
        z_s, _ = self.encoder.encoder(x_s, edge_index_s, edge_attr_s, batch_s)
        z_t, _ = self.encoder.encoder(x_t, edge_index_t, edge_attr_t, batch_t)
        z_s, s_mask = to_dense_batch(z_s, batch_s, fill_value=0) # z_s: batch_size * num_node_max * channel. s_mask: batch_size * num_node_mask.
        z_t, t_mask = to_dense_batch(z_t, batch_t, fill_value=0)

        s_num_node = s_mask.sum(dim=-1)   
        t_num_node = t_mask.sum(dim=-1)
        
        assert z_s.size(0) == z_t.size(0), 'Encountered unequal batch-sizes'

        
        for k in range(self.__iteration):
            if k == 0:
                # unary_affs = self.affinity_layer(z_s, z_t)
                unary_affs = torch.matmul(z_s, z_t.transpose(-2, -1))
            else:
                # unary_affs = self.alpha_1 * self.affinity_layer(z_s, z_t) + self.alpha_2 * self.affinity_layer(torch.bmm(binary_m, z_t), torch.bmm(binary_m.transpose(1, 2), z_s))
                unary_affs = self.alpha_1 * torch.matmul(z_s, z_t.transpose(-2, -1)) + self.alpha_2 * torch.matmul(torch.bmm(binary_m, z_t), torch.bmm(binary_m.transpose(1, 2), z_s).transpose(-2, -1))
            
            probability = self.voting_layer(unary_affs, s_num_node, t_num_node)
            double_stochastic_m = self.bi_stochastic(probability, s_num_node, t_num_node, dummy_row=True)
            binary_m = self.hungarian(double_stochastic_m, s_num_node, t_num_node)

        return probability, binary_m, s_num_node, t_num_node

    def forward_ori(self, data_dict):
        graphs = data_dict['pyg_graphs']
        num_graphs = len(graphs)
        n_points = data_dict['ns']

        assert 'orig_graph_list' in data_dict, 'There are no orig_graph_list, please run the pre_train model at first.'
        orig_graph_list = data_dict['orig_graph_list']
        
        for k in range(self.__iteration):
            if k == 0:
                unary_affs_list = [
                    self.affinity_layer([item.x for item in g_1], [item.x for item in g_2])
                    for (g_1, g_2) in lexico_iter(orig_graph_list)
                ]
                for unary_affs, (idx1, idx2) in zip(unary_affs_list, lexico_iter(range(num_graphs))):
                    unary_affs = torch.stack(pad_tensor(unary_affs), dim = 0)
                    probability = self.voting_layer(unary_affs, n_points[idx1], n_points[idx2])
                    double_stochastic_m = self.bi_stochastic(probability,  n_points[idx1], n_points[idx2])
                    binary_m = self.hungarian(double_stochastic_m,  n_points[idx1], n_points[idx2])
   
            else:
                unary_affs_oris = [self.affinity_layer([item.x for item in g_1], [item.x for item in g_2]) for (g_1, g_2) in lexico_iter(orig_graph_list)]
                unary_affs_ias = [
                    self.affinity_layer(
                        torch.bmm(binary_m, torch.stack(pad_tensor([item.x for item in g_2]),  dim = 0)), 
                        torch.bmm(binary_m.transpose(1, 2), torch.stack(pad_tensor([item.x for item in g_1]), dim = 0))) 
                        for (g_1, g_2) in lexico_iter(orig_graph_list)
                ] 
                # * len(unary_affs_oris) = len(unary_affs_ias[0]) = batch_size

                for unary_affs_ori, unary_affs_ia, (idx1, idx2) in zip(unary_affs_oris, unary_affs_ias, lexico_iter(range(num_graphs))):
                    unary_affs = self.a1 * torch.stack(pad_tensor(unary_affs_ori), dim = 0) + self.a2 * torch.stack(pad_tensor(unary_affs_ia), dim = 0)
                    probability = self.voting_layer(unary_affs, n_points[idx1], n_points[idx2])
                    double_stochastic_m = self.bi_stochastic(probability,  n_points[idx1], n_points[idx2])
                    binary_m = self.hungarian(double_stochastic_m,  n_points[idx1], n_points[idx2])
            
            data_dict.update({
                's_mat': probability,
                'ds_mat': double_stochastic_m,
                'perm_mat': binary_m,
            })
        
        return data_dict

                  
class Matching_cross_attention(nn.Module):
    def __init__(self):
        super(Matching_cross_attention, self).__init__()
        self.encoder = GCL()

        if len(cfg.PRETRAINED_PATH) > 0:
            encoder_model_path = cfg.PRETRAINED_PATH
            print('Loading model from {}'.format(encoder_model_path))
            load_model(self.encoder, encoder_model_path, strict=False)
        if cfg.TRAIN.BACKBONE_LR == 0.0:
            for p in self.encoder.parameters():
                p.requires_grad = False
        
        self.voting_layer = FullPro()
        self.bi_stochastic = Sinkhorn()
        self.hungarian = hungarian

        self.alpha_1 = cfg.BIIA.ALPHA1
        self.alpha_2 = cfg.BIIA.ALPHA2
        # pseudo
        self.__iteration = cfg.BIIA.ITERATION_
    
    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s, x_t, edge_index_t, edge_attr_t, batch_t):
        z_s, _ = self.encoder.encoder(x_s, edge_index_s, edge_attr_s, batch_s)
        z_t, _ = self.encoder.encoder(x_t, edge_index_t, edge_attr_t, batch_t)

        # z_s = self.encoder.project(z_s)
        # z_t = self.encoder.project(z_t)

        z_s, s_mask = to_dense_batch(z_s, batch_s, fill_value=0)
        z_t, t_mask = to_dense_batch(z_t, batch_t, fill_value=0)

        s_num_node = s_mask.sum(dim=-1)
        t_num_node = t_mask.sum(dim=-1)
        (batch_size, N_s, _), N_t = z_s.size(), z_t.size(1)

        assert z_s.size(0) == z_t.size(0), 'Encountered unequal batch-sizes'

        unary_affs = torch.matmul(z_s, z_t.transpose(-2, -1))
        probability = self.voting_layer(unary_affs, s_num_node, t_num_node)
        # cross_double_stochastic_m = double_stochastic_m = self.bi_stochastic(probability, s_num_node, t_num_node, dummy_row=True)
        cross_binary_m = binary_m = self.hungarian(probability, s_num_node, t_num_node)
        
        # last_cross_z_s = z_s # B X N_s X Feature
        # last_cross_z_t = z_t
        last_cross_z_s = to_dense(x_s, s_mask)
        last_cross_z_t = to_dense(x_t, t_mask)
        # double_stochastic_m_mask = s_mask.view(batch_size, N_s, 1) & t_mask.view(batch_size, 1, N_t)   

        for k in range(self.__iteration):
            cross_z_s = to_sparse(torch.bmm(cross_binary_m, last_cross_z_t), s_mask) # (N1+N2+...) x Feature
            cross_z_t = to_sparse(torch.bmm(cross_binary_m.transpose(1, 2), last_cross_z_s), t_mask) 
            cross_z_s, _ = self.encoder.encoder(cross_z_s, edge_index_s, edge_attr_s, batch_s)
            cross_z_t, _ = self.encoder.encoder(cross_z_t, edge_index_t, edge_attr_t, batch_t)
            cross_z_s, cross_z_t = to_dense(cross_z_s, s_mask), to_dense(cross_z_t, t_mask)

            assert cross_z_s.size(0) == cross_z_s.size(0), 'Encountered unequal batch-sizes'

            cross_unary_affs = torch.matmul(cross_z_s, cross_z_t.transpose(-2, -1))
            cross_probability = self.voting_layer(cross_unary_affs, s_num_node, t_num_node)
            cross_double_stochastic_m = self.bi_stochastic(cross_probability, s_num_node, t_num_node, dummy_row=True)

            cross_probability_sum = self.voting_layer(cross_probability + probability, s_num_node, t_num_node)         
            # cross_double_stochastic_m_sum = cross_double_stochastic_m + double_stochastic_m
            cross_double_stochastic_m_sum = self.bi_stochastic(cross_probability_sum, s_num_node, t_num_node, dummy_row=True)
            cross_binary_m_sum = self.hungarian(cross_double_stochastic_m_sum, s_num_node, t_num_node)
        
            

        return probability,  binary_m, s_num_node, t_num_node, probability, cross_probability


                
