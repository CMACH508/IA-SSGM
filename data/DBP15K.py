import enum
import os
import os.path as osp
import shutil
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.utils.rnn as rnn
from torch_geometric.data import Data, InMemoryDataset

from torch_geometric.io import read_txt_array
from torch_geometric.utils import sort_edge_index

class DBP15K(InMemoryDataset):
    r"""The DBP15K dataset from the
    `"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding"
    <https://arxiv.org/abs/1708.05045>`_ paper, where Chinese, Japanese and
    French versions of DBpedia were linked to its English version.
    Node features are given by pre-trained and aligned monolingual word
    embeddings from the `"Cross-lingual Knowledge Graph Alignment via Graph
    Matching Neural Network" <https://arxiv.org/abs/1905.11605>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        pair (string): The pair of languages (:obj:`"en_zh"`, :obj:`"en_fr"`,
            :obj:`"en_ja"`, :obj:`"zh_en"`, :obj:`"fr_en"`, :obj:`"ja_en"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    pairs = ['en_zh', 'en_fr', 'en_ja', 'zh_en', 'fr_en', 'ja_en']
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
            for i, line in enumerate(f):
                info = line.strip().split(' ')
                if len(info) > 300:
                    embs[info[0]] = torch.tensor([float(x) for x in info[1:]])
                else:
                    embs['**UNK**'] = torch.tensor([float(x) for x in info])
        
        # rel_ids_1: ids for relationships in source KG (ZH);
        # rel_ids_2: ids for relationships in target KG (EN);
        # triples_1: relationship triples encoded by ids in source KG (ZH);
        # triples_2: relationship triples encoded by ids in target KG (EN);
        g1_path = osp.join(self.raw_dir, self.pair, 'triples_1')
        x1_path = osp.join(self.raw_dir, self.pair, 'id_features_1')
        g2_path = osp.join(self.raw_dir, self.pair, 'triples_2')
        x2_path = osp.join(self.raw_dir, self.pair, 'id_features_2')
        
        x1, edge_index1, rel1, assoc1 = self.process_graph(
            g1_path, x1_path, embs)
        x2, edge_index2, rel2, assoc2 = self.process_graph(
            g2_path, x2_path, embs)
        
        train_path = osp.join(self.raw_dir, self.pair, 'train.examples.20')
        train_y = self.process_y(train_path, assoc1, assoc2)
        
        test_path = osp.join(self.raw_dir, self.pair, 'test.examples.1000')
        test_y = self.process_y(test_path, assoc1, assoc2)
        data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1, x2=x2, 
                    edge_index2=edge_index2, rel2=rel2, train_y=train_y,
                    test_y=test_y)
        torch.save(self.collate([data]), self.processed_paths[0])
        
    
    def process_graph(
        self, 
        triple_path: str, 
        feature_path: str, 
        embeddings: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    
        g1 = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        subj, rel, obj = g1.t()
        
        x_dict = {}
        with open(feature_path, 'r') as f:
            for line in f:
                info = line.strip().split('\t') 
                info = info if len(info) == 2 else info + ['**UNK**']
                seq = info[1].lower().split() # seq: entity
                hs = [embeddings.get(w, embeddings['**UNK**']) for w in seq] # dict.get(key[, value]) value -- 可选，如果指定键的值不存在时，返回该默认值
                x_dict[int(info[0])] = torch.stack(hs, dim=0)
        
        idx = torch.tensor(list(x_dict.keys())) # x_dict.key: info[0] 也就 entity 的id
        assoc = torch.full((idx.max().item() + 1, ), -1, dtype=torch.long)
        assoc[idx] = torch.arange(idx.size(0))  
        # 这里 assoc 相当于对 id_features_1(id_features_2)中的 entity id 与他在文件中出现的顺序做了一个对应

        subj, obj = assoc[subj], assoc[obj]  # id_feature_1 中的entity id 变换为文件顺序id，也可以理解为矩阵的行列id
        edge_index = torch.stack([subj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index=edge_index, edge_attr=rel)
        
        xs = [None for _ in range(idx.size(0))]
        for i in x_dict.keys():
            xs[assoc[i]] = x_dict[i]
    
        x = rnn.pad_sequence(xs, batch_first=True)
        
        return x, edge_index, rel, assoc
    
    def process_y(self, path: str, assoc1: Tensor, assoc2: Tensor) -> Tensor:
        row, col, mask = read_txt_array(path, sep='\t', dtype=torch.long).t()
        mask = mask.to(torch.bool)

        return torch.stack([assoc1[row[mask]], assoc2[col[mask]]], dim=0)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.pair})'

class SumEmbedding(object):
    def __call__(self, data):
        # 把entity 的word embeddings 加起来，比如 James Bob 有俩个words, 把他们对应的embedding 加起来
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data



if __name__ == '__main__':
    path = osp.join('..', 'data', 'DBP15K_test')
    for pair in DBP15K.pairs:
        data = DBP15K(path, pair, transform=SumEmbedding())[0]

        
        
        
