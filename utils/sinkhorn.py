import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

class Sinkhorn(nn.Module):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=20, tau: float=1., epsilon=1e-4, batched_operation=False):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.batched_operation = batched_operation # batched operation may cause instability in backward computation,
                                                   # but will boost computation.
        self.tau = tau

    def forward(self, s, nrows=None, ncols=None, dummy_row=False, dtype=torch.float32):
        """
        Parameters: dummy_row: 选择是否要把每一个batch的矩阵尺寸大小设置为一致 
        """
        # computing sinkhorn with row/column normalization in the log space.
        if len(s.shape) == 2:
            s = s.unsqueeze(0) # 在0纬度扩充一维度
            matrix_input = True #标注matrix是否做了padding
        elif len(s.shape) == 3:
            matrix_input = False 
        else:
            raise ValueError('input data shape not understood.')
        
        batch_size = s.shape[0]
        
        # 保证每一对sample的双随机矩阵的行数<列数
        if s.shape[2] >= s.shape[1]:
            transposed = False
        else:
            s = s.transpose(1, 2)
            transposed = True
            ori_nrows = nrows
            nrows = ncols
            ncols = ori_nrows
        

        # model 中是给出的, 这块不运行
        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)] #列表形式记录这一个batch每个双随机矩阵的行数， 记住: 这一个batch的行都是一样的
            print(nrows)
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]
            print(ncols)

        # operations are performed on log_s
        s = s / self.tau
        
        if dummy_row:
            assert s.shape[2] >= s.shape[1]
            dummy_shape = list(s.shape) # dummy_shape=[batch_size, n1_gt, n2_gt]
            dummy_shape[1] = s.shape[2] - s.shape[1] # dummy_shape[1] = n2_gt - n1_gt
            ori_nrows = nrows
            nrows = ncols
            s = torch.cat((s, torch.full(dummy_shape, -float('inf')).to(s.device)), dim=1) # torch.full() create了一个size=dummy_shape 的矩阵，
                                                                                           # s 扩充成为了[batch_size, n2_gt, n2_gt]的矩阵，扩充部分的值为-inf, 负无穷
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100 # 扩充的那几行，padding值为-100
                s[b, nrows[b]:, :] = -float('inf') # 为了照顾同一个batch的其他sample，每一个双随机矩阵的size的行和列可能大于这一对sample的max_node_num
                s[b, :, ncols[b]:] = -float('inf') # 这部分填充为负无穷
        
        if self.batched_operation: # 批量操作
            log_s = s
            
            for i in range(self.max_iter):
                if i % 2 == 0:
                    # row norm
                    log_sum = torch.logsumexp(log_s, 2, keepdim=True) # 双随机矩阵的按照 每行的元素e^x后求和，再做log操作，
                                                                      # keepdim output tensor is of the same size as input except in the dimension(s) dim where it is of size 1.
                    log_s = log_s - log_sum                           # 每个元素减去求出的log和，log相减，相当于是做除法
                    log_s[torch.isnan(log_s)] = -float('inf')         # torch.isnan()返回一个tensor判断每个元素是不是nan, 将元素为nan的值置为负无穷
                else:
                    # column norm
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)  
                    log_s = log_s - log_sum
                    log_s[torch.isnan(log_s)] = -float('inf')
                
                if dummy_row and dummy_shape[1] > 0:      # 如果扩充了双随机矩阵的行，
                    log_s = log_s[:, :-dummy_shape[1]]    # 就把扩充的那一行删掉
                    for b in range(batch_size):
                        log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf') # TODO: # 正常情况下是无效操作？
                
                if matrix_input:         # 之前扩充成3维的矩阵，降为2维
                    log_s.squeeze_(0)
                
                return torch.exp(log_s)
        
        else:
            ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype) # dummy_row 设置为True的话，ret_log_s.shape = [batch_size, n_cols, n_rows]
            
            for b in range(batch_size):
                row_slice = slice(0, nrows[b]) 
                col_slice = slice(0, ncols[b])
                log_s = s[b, row_slice, col_slice]

                for i in range(self.max_iter):
                    if i % 2 == 0:
                        log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                        log_s = log_s - log_sum
                    else:
                        log_sum = torch.logsumexp(log_s, 0, keepdim=True)
                        log_s = log_s - log_sum
            
                ret_log_s[b, row_slice, col_slice] = log_s

            if dummy_row:
                if dummy_shape[1] > 0:
                    ret_log_s = ret_log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if transposed:
                ret_log_s = ret_log_s.transpose(1, 2)
            if matrix_input:
                ret_log_s.squeeze_(0)

            return torch.exp(ret_log_s)

class Sinkhorn_sparse(nn.Module):
    def __init__(self, max_iter=10):
        super(Sinkhorn_sparse, self).__init__()
        self.max_iter = max_iter
    
    def forward(self, sims):
        s = torch.exp(sims*50)
        num_row, num_col = s.shape
        device = sims.device

        for k in range(self.max_iter):
            s = s / torch.sum(s,dim=1,keepdims=True)
            s = s / torch.sum(s,dim=0,keepdims=True)
        
        row = torch.tensor([i for i in range(min(num_row, num_col))]).to(device)
        col = torch.argmax(s, dim=-1)
        col = col[: min(num_row, num_col)]

        indices = torch.stack((row, col), dim=0).to(device)
        values = torch.ones(len(row)).to(device)
        return torch.sparse_coo_tensor(indices=indices, values=values, size=[s.size(0),s.size(1)], device=device)

    
if __name__ == '__main__':
    bs = Sinkhorn(max_iter=8, epsilon=1e-4)
    inp = torch.tensor([[[1., 0, 1.,0],
                         [1., 0, 3.,0],
                         [2., 0, 1.,0],
                         [0, 0, 0, 0]
                         ]], requires_grad=True)
    print (inp.shape)
    outp = bs(inp,[4],[3], dummy_row=True)

    print(outp)
    l = torch.sum(outp)
    outp.retain_grad()
    l.backward()
   
    print(inp.grad )

    outp2 = torch.tensor([[0.1, 0.1, 1],
                          [2, 3, 4.]], requires_grad=True)

    l = torch.sum(outp2)
    l.backward()
    print(outp2.grad)

# import torch
# import torch.nn as nn


# class Sinkhorn(nn.Module):
#     """
#     BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
#     Parameter: maximum iterations max_iter
#                a small number for numerical stability epsilon
#     Input: input matrix s
#     Output: bi-stochastic matrix s
#     """
#     def __init__(self, max_iter=10, epsilon=1e-4):
#         super(Sinkhorn, self).__init__()
#         self.max_iter = max_iter
#         self.epsilon = epsilon

#     def forward(self, s, nrows=None, ncols=None, exp=False, exp_alpha=20, dummy_row=False, dtype=torch.float32):
#         batch_size = s.shape[0]

#         if dummy_row:
#             dummy_shape = list(s.shape)
#             dummy_shape[1] = s.shape[2] - s.shape[1]
#             s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
#             new_nrows = ncols
#             for b in range(batch_size):
#                 s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
#             nrows = new_nrows

#         row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device)  # size: row x row
#         col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device)  # size: col x col
#         for b in range(batch_size):
#             row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
#             col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
#             row_norm_ones[b, row_slice, row_slice] = 1
#             col_norm_ones[b, col_slice, col_slice] = 1

#         # for Sinkhorn stacked on last dimension
#         if len(s.shape) == 4:
#             row_norm_ones = row_norm_ones.unsqueeze(-1)
#             col_norm_ones = col_norm_ones.unsqueeze(-1)

#         s += self.epsilon

#         for i in range(self.max_iter):
#             if exp:
#                 s = torch.exp(exp_alpha * s)
#             if i % 2 == 1:
#                 # column norm
#                 sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
#             else:
#                 # row norm
#                 sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

#             tmp = torch.zeros_like(s)
#             for b in range(batch_size):
#                 row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
#                 col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
#                 tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
#             s = s * tmp

#         if dummy_row and dummy_shape[1] > 0:
#             s = s[:, :-dummy_shape[1]]

#         return s
