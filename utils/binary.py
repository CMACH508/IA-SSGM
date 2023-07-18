import torch
import torch.nn as nn
import numpy as np

class Binary(nn.Module):
    """
    Binary Layer turns the input bi-stochastic matrix into a 0,1 matrix.
    Input: input matrix s
    Output: 0,1 perm_mat
    """
    def __init__(self):
        super(Binary, self).__init__()
    
    def forward(self, s: torch.Tensor, n1=None, n2=None):
        s_tmp = s.clone()
        # s_tmp = s_tmp + 1e-20
        device = s.device
        perm_mat = torch.zeros_like(s)
        for b in range(s.shape[0]):
            n1b = s.shape[1] if n1 is None else n1[b]
            n2b = s.shape[2] if n2 is None else n2[b]
            for _ in range(n1b):
                index = s_tmp[b, :n1b, :n2b].argmax()
                rowindex = index // n1b
                colindex = index % n1b
                s_tmp[b, rowindex, :n2b] = 0
                s_tmp[b, :n1b, colindex] = 0
                perm_mat[b][rowindex][colindex] = 1
        perm_mat = perm_mat.to(device)
        return perm_mat