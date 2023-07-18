from optparse import Values
import torch
import scipy.optimize as opt
import numpy as np


def hungarian(s: torch.Tensor, n1=None, n2=None):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :return: optimal permutation matrix
    """
    if len(s.shape) == 2:
        s = s.unsqueeze(0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False

    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    for b in range(batch_num):
        n1b = perm_mat.shape[1] if n1 is None else n1[b]
        n2b = perm_mat.shape[2] if n2 is None else n2[b]
        row, col = opt.linear_sum_assignment(perm_mat[b, :n1b, :n2b])
        perm_mat[b] = np.zeros_like(perm_mat[b])
        perm_mat[b, row, col] = 1
    perm_mat = torch.from_numpy(perm_mat).to(device)

    if matrix_input:
        perm_mat.squeeze_(0)
        
    return perm_mat

def hungarian_sparse(s: torch.Tensor):
    device = s.device

    res = s.cpu().detach().numpy() * -1
    if len(res.shape) != 2:
        res = res.squeeze(0)
    row, col = opt.linear_sum_assignment(res)
    indices = np.stack((row, col), axis=0)
    values = torch.ones(len(row))
    
    return torch.sparse_coo_tensor(indices=torch.tensor(indices), values=values, size=[s.size(0),s.size(1)], device=device)

