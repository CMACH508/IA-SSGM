import torch
import numpy as np
import torch.nn.functional as F

def pad_tensor(inp):
    """ Pad a list of tensor(may be in different dimension) into a list of tensor with the same dimension
        Paramter:
        inp: a list of tensor
        Return:
        padded_ts: a list of tensor with the same dimension.
    """
    assert type(inp[0]) == torch.Tensor
    it = iter(inp)
    t = next(it)
    max_shape = list(t.shape)
    while True:
        try:
            t = next(it)
            for i in range(len(max_shape)):
                max_shape[i] = int(max(max_shape[i], t.shape[i]))
        except StopIteration:
            break

    max_shape = np.array(max_shape)
    padded_ts = []
    for t in inp:
        pad_pattern = np.zeros(2 * len(max_shape), dtype= np.int64) # 对inp的每个dimension都做填充
        pad_pattern[::-2] = max_shape - np.array(t.shape)
        pad_pattern = tuple(pad_pattern.tolist())
        padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))
        
    return padded_ts


