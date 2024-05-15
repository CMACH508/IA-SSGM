import enum
import os
import os.path as osp
import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.utils.rnn as rnn
from torch_geometric.data import Data, InMemoryDataset

from torch_geometric.io import read_txt_array
from torch_geometric.utils import sort_edge_index

class DBP15K(InMemoryDataset):
    pairs = ['fr_en', 'ja_en', 'zh_en']
    def __init__(self, root:str, pair: str, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        assert pair in ['en_zh', 'en_fr', 'en_ja', 'zh_en', 'fr_en', 'ja_en']
        self.pair = pair
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def raw_file_names(self) -> List[str]:
        return ['en_zh', 'en_fr', 'en_ja', 'zh_en', 'fr_en', 'ja_en']
    
    @property
    def processed_file_names(self) -> str:
        return f'{self.pair}.pt'
    
    def process(self):
        embs = {}
        # sub.glove.300d, 序列化用GloVE 预处理的word 的feature 
        with open(osp.join(self.raw_dir, 'sub.glove.300d'), 'r') as f:
            for _, line in enumerate(f):
                info = line.strip().split(' ')
                if len(info) > 300:
                    embs[info[0]] = torch.tensor([float(x) for x in info[1:]])
                else:
                    embs['**UNK**'] = torch.tensor([float(x) for x in info])
        
        g1_path = osp.join(self.raw_dir, self.pair, 'triples_1')
        x1_path = osp.join(self.raw_dir, self.pair, 'id_features_1')
        g2_path = osp.join(self.raw_dir, self.pair, 'triples_2')
        x2_path = osp.join(self.raw_dir, self.pair, 'id_features_2')
        
        x1, edge_index1, rel1, assoc1 = self.process_graph(
            g1_path, x1_path, embs)
        x2, edge_index2, rel2, assoc2 = self.process_graph(
            g2_path, x2_path, embs)

        gt_path = osp.join(self.raw_dir, self.pair, 'ref_ent_ids')
        gt_y = self.process_y(gt_path, assoc1, assoc2)

        data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1, x2=x2, 
                    edge_index2=edge_index2, rel2=rel2, gt_y=gt_y)

        torch.save(self.collate([data]), self.processed_paths[0])

    def process_graph(self, triple_path: str, entity_feature_path: str, embeddings: Dict[str, Tensor],) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        :Params: triple_path, relationship triples encoded by ids in source (target) KG. can be considered as edge index.
        :Params: feature_path, translated entity name.
        :Params: word embeddings dicts.
        """
        g1 = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        subj, rel, obj = g1.t()

        # load traslated entity name.
        x_dict = {}
        with open(entity_feature_path, 'r') as f:
            for line in f:
                info = line.strip().split('\t') 
                info = info if len(info) == 2 else info + ['**UNK**']
                seq = info[1].lower().split() # seq: entity
                hs = [embeddings.get(w, embeddings['**UNK**']) for w in seq] # dict.get(key[, value]) value -- 可选，如果指定键的值不存在时，返回该默认值
                x_dict[int(info[0])] = torch.stack(hs, dim=0)
        
        idx = list(x_dict.keys())
        random.shuffle(idx)
        # generate mapping that entity_id to nodes_order.
        assoc = torch.full((max(idx) + 1, ), -1, dtype=torch.long)
        assoc[idx] = torch.arange(len(idx))

        subj, obj = assoc[subj], assoc[obj]  # id_feature_1 中的entity id 变换为文件顺序id，也可以理解为矩阵的行列id
        edge_index = torch.stack([subj, obj], dim=0)
        
        edge_index, rel = sort_edge_index(edge_index=edge_index, edge_attr=rel)
        

        # generate nodes feature.
        xs = [None for _ in range(len(idx))]
        for i in idx:
            xs[assoc[i]] = x_dict[i]
        
        x = rnn.pad_sequence(xs, batch_first=True) # shape: len(idx) * 300
        
        # # if translated relation file exists
        # if osp.exists(relation_feature_path):
        #     rel_dict = {}
        #     with open(relation_feature_path, 'r') as f:
        #         for line in f:
        #             info = line.strip().split('\t') 
        #             info = info if len(info) == 2 else info + ['**UNK**']
        #             seq = info[1].lower().split() # seq: entity
        #             hs = [embeddings.get(w, embeddings['**UNK**']) for w in seq] # dict.get(key[, value]) value -- 可选，如果指定键的值不存在时，返回该默认值
        #             rel_dict[int(info[0])] = torch.stack(hs, dim=0)
            
        #     rels = [None for _ in range(rel.shape[0])]
        #     for i in range(rel.shape[0]):
        #         rels[i] = rel_dict[rel[i].item()]
                
        #     rel = rnn.pad_sequence(rels, batch_first=True)
        #     print(edge_index.shape, rel.shape)
 
        return x, edge_index, rel, assoc

    def process_y(self, path: str, assoc1: Tensor, assoc2: Tensor) -> Tensor:
        row, col = read_txt_array(path, sep='\t', dtype=torch.long).t()       
        return torch.stack([assoc1[row], assoc2[col]], dim=0)

class SumEmbedding(object):
    def __call__(self, data):
        # 把entity 的word embeddings 加起来，比如 James Bob 有俩个words, 把他们对应的embedding 加起来
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        
        return data

if __name__ == '__main__':
    path = osp.join('..', 'data', 'DBP15K_SSL')
    for pair in DBP15K.pairs:
        data = DBP15K(path, pair, transform=SumEmbedding())[0]
    
