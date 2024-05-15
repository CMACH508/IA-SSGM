import re
from itertools import chain

import torch
import random
import numpy as np
from torch.utils import data
from torch_geometric.data import Data
import torch_geometric.transforms as T
from utils.config import cfg
from torch.utils.data import Dataset

from data.full_connected import FullConnected


class PairData(Data):  # pragma: no cover
    def __inc__(self, key, value,  *args):
        if bool(re.search('index_s', key)):
            return self.x_s.size(0)
        if bool(re.search('index_t', key)):
            return self.x_t.size(0)
        else:
            return 0


class PairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building pairs between separate dataset examples.

    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
    """
    def __init__(self, dataset_s, dataset_t, sample=False):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.sample = sample

    def __len__(self):
        return len(self.dataset_s) if self.sample else len(
            self.dataset_s) * len(self.dataset_t)

    def __getitem__(self, idx):
        if self.sample:
            data_s = self.dataset_s[idx]
            data_t = self.dataset_t[random.randint(0, len(self.dataset_t) - 1)]
        else:
            data_s = self.dataset_s[idx // len(self.dataset_t)]
            data_t = self.dataset_t[idx % len(self.dataset_t)]

        return PairData(
            x_s=data_s.x,
            edge_index_s=data_s.edge_index,
            edge_attr_s=data_s.edge_attr,
            x_t=data_t.x,
            edge_index_t=data_t.edge_index,
            edge_attr_t=data_t.edge_attr,
            num_nodes=None,
        )

    def __repr__(self):
        return '{}({}, {}, sample={})'.format(self.__class__.__name__,
                                              self.dataset_s, self.dataset_t,
                                              self.sample)


class ValidPairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by 
    building valid pairs between separate dataset examples.
    A pair is valid if each node class in the source graph also exit in the 
    target graph.
    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
    """
    def __init__(self, dataset_s, dataset_t, sample=True, random=False):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.pairs, self.cumdeg, self.random_index_list = self.__compute_pairs__()
        self.sample = sample
        self.random = random # self.random = True, will randomly sample from all self.pairs.
    
    def __compute_pairs__(self):
        num_keypoints = 0
        for data in chain(self.dataset_s, self.dataset_t):
            num_keypoints = max(num_keypoints, data.y.max().item() + 1)
        
        y_s = torch.zeros((len(self.dataset_s), num_keypoints), dtype=torch.bool)
        y_t = torch.zeros((len(self.dataset_t), num_keypoints), dtype=torch.bool)
        # one-hot for keypoints_name.
        for i, data in enumerate(self.dataset_s):
            y_s[i, data.y] = 1
        for i, data in enumerate(self.dataset_t):
            y_t[i, data.y] = 1
        
        y_s = y_s.view(len(self.dataset_s), 1, num_keypoints)
        y_t = y_t.view(1, len(self.dataset_t), num_keypoints)

        pairs = ((y_s * y_t).sum(dim=-1) == y_s.sum(dim=-1)).nonzero() # len(self.dataset_s) * len(self.dataset_t)
        cumdeg = pairs[:, 0].bincount().cumsum(dim=0)

        random_index_list = np.random.randint(low=0, high=len(pairs), size=200)
        return pairs.tolist(), [0] + cumdeg.tolist(), random_index_list.tolist()
    
    def __len__(self):
        if self.sample:
            return len(self.dataset_s)
        elif self.random:
            return len(self.random_index_list)
        else:
            return len(self.pairs)
    
    def __getitem__(self, idx):
        if self.sample:
            data_s = self.dataset_s[idx]
            i = random.randint(self.cumdeg[idx], self.cumdeg[idx + 1] - 1)
            data_t = self.dataset_t[self.pairs[i][1]]
        elif self.random:
            data_s = self.dataset_s[self.pairs[self.random_index_list[idx]][0]]
            data_t = self.dataset_t[self.pairs[self.random_index_list[idx]][1]]
        else:
            data_s = self.dataset_s[self.pairs[idx][0]]
            data_t = self.dataset_t[self.pairs[idx][1]]
        
        y = data_s.y.new_full((data_t.y.max().item() + 1, ), -1) # 返回一个和data_s.y 一样的dtype, device 的tensor.
        y[data_t.y] = torch.arange(data_t.num_nodes)
        y = y[data_s.y] # y 表示，x_s，对应在y_s的序列6
        

        return PairData(
            x_s=data_s.x,
            edge_index_s=data_s.edge_index,
            edge_attr_s=data_s.edge_attr,
            img_s=data_s.img if data_s.img is not None else None,
            pos_s=data_s.pos if data_s.pos is not None else None,
            name_s=data_s.name if data_s.name is not None else None,
            y_s=data_s.y if data_s.y is not None else None,
            x_t=data_t.x,
            edge_index_t=data_t.edge_index,
            edge_attr_t=data_t.edge_attr,
            img_t=data_t.img if data_t.img is not None else None,
            pos_t=data_t.pos if data_t.pos is not None else None,
            name_t=data_t.name if data_t.name is not None else None,
            y_t=data_t.y if data_t.y is not None else None,
            y=y,
            num_nodes=None,
        )


    def __repr__(self):
        return '{}({}, {}, sample={})'.format(self.__class__.__name__,
                                              self.dataset_s, self.dataset_t,
                                              self.sample)

class SamePairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building same pairs between separate dataset examples.
    A pair is Same if the number of node in the source graph is the same with 
    target graph.
    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
    """
    def __init__(self, dataset_s, dataset_t, train=False, random=True):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.train = train
    
    def __len__(self):
        if self.train:
            return cfg.TRAIN.EPOCH_ITERS
        else:
            return cfg.EVAL.SAMPLES

    def __getitem__(self, idx):
        data_s, data_t, y =  self.__compute_pairs__(idx)
        
        return PairData(
            x_s=data_s.x,
            edge_index_s=data_s.edge_index,
            edge_attr_s=data_s.edge_attr,
            img_s=data_s.img if data_s.img is not None else None,
            pos_s=data_s.pos if data_s.pos is not None else None,
            name_s=data_s.name if data_s.name is not None else None,
            y_s=data_s.y if data_s.y is not None else None,
            x_t=data_t.x,
            edge_index_t=data_t.edge_index,
            edge_attr_t=data_t.edge_attr,
            img_t=data_t.img if data_t.img is not None else None,
            pos_t=data_t.pos if data_t.pos is not None else None,
            name_t=data_t.name if data_t.name is not None else None,
            y_t=data_t.y if data_t.y is not None else None,
            y=y,
            num_nodes=None,
        )

    def __compute_pairs__(self, idx):
        while True:
            data_s = random.choice(self.dataset_s)
            data_t = random.choice(self.dataset_t)
            # num_keypoints = 0
            # for data in chain(self.dataset_s, self.dataset_t):
            #     num_keypoints = max(num_keypoints, data.y.max().item() + 1)
            
            y = data_s.y.new_full((data_s.y.max().item() + 1, data_t.y.max().item() + 1), 0)
            row_list, col_list = [], []
            for i, keypoint in enumerate(data_s.y):
                for j, _keypoint in enumerate(data_t.y):
                    if keypoint == _keypoint:
                        y[i, j] = 1
                        row_list.append(i)
                        col_list.append(j)
                        break 
            row_list.sort()
            col_list.sort()
            y = y[row_list, :]
            y = y[:, col_list]
            
            y = torch.nonzero(y, as_tuple=True)[1]
            
            data_s = Data(img=data_s.img, x = data_s.x[row_list, :], pos=data_s.pos[row_list, :], name=data_s.name, y=data_s.y[row_list])
            data_t = Data(img=data_t.img, x = data_t.x[col_list, :], pos=data_t.pos[col_list, :], name=data_t.name, y=data_t.y[col_list])
            # data_t = Data(img=data_t.img, x = data_t.x, pos=data_t.pos, name=data_t.name, y=data_t.y)
            
            filter = lambda data: data.pos.size(0) > 1
            if filter(data_s) and filter(data_t):
                transform = T.Compose([
                    T.Delaunay(),
                    T.FaceToEdge(),
                    T.Cartesian(),
                ])
                # transform = T.Compose([
                #     FullConnected(),
                #     T.Cartesian(),
                # ])
                break
                
        return transform(data_s), transform(data_t), y


        
        

        
       