import torch

from torch_geometric.utils import dense_to_sparse
from torch_geometric.transforms import BaseTransform


class FullConnected(BaseTransform):
    r"""Creates a full-connected graph.
a

    """
    def __init__(self, add_self_loops: bool = False):
        self.add_self_loops = add_self_loops
    
    def __call__(self, data):
        if data.pos.size(0) == 1:
            data.edge_index = torch.tensor([], dtype=torch.long,
                                           device=data.pos.device).view(2, 0)
        if data.pos.size(0) > 1:
            if not self.add_self_loops:
                adj = torch.ones(data.pos.size(0), data.pos.size(0)) - torch.eye(data.pos.size(0))
            else:
                adj = torch.ones(data.pos.size(0), data.pos.size(0))

            edge_index, _ = dense_to_sparse(adj)
            data.edge_index = edge_index
            
        return data


            
