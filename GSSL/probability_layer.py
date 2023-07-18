import torch
import torch.nn as nn
import torch.nn.functional as F

class FullPro(nn.Module):
    def __init__(self, alpha=20, pixel_thresh=None):
        super(FullPro, self).__init__()
        # self.batch_size = batch_size
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)
        self.pixel_thresh = pixel_thresh

    def forward(self, s, nrow_gt, ncol_gt=None):
        norm_s = F.normalize(s, p = 2, dim = -1)
        ret_s = torch.zeros_like(s)
        # filter dummy nodes
        for b, n in enumerate(nrow_gt):
            if ncol_gt is None:
                ret_s[b, 0:n, :] = self.softmax(norm_s[b, 0:n, :])
                    # self.softmax(torch.mul(W1[b, 0:n, :], self.alpha * s[b, 0:n, :]))
            else:
                ret_s[b, 0:n, 0:ncol_gt[b]] =self.softmax(norm_s[b, 0:n, 0:ncol_gt[b]])
                    # self.softmax(torch.mul(W1[b, 0:n, 0:ncol_gt[b]], self.alpha * s[b, 0:n, 0:ncol_gt[b]]))

        return ret_s