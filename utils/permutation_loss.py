import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)

        try:
            assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_perm)
            raise err

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        
        for b in range(batch_num):
            batch_slice = [b, slice(pred_ns[b]), slice(gt_ns[b])]
            loss += F.binary_cross_entropy(
                pred_perm[batch_slice],
                gt_perm[batch_slice],
                reduction='sum')
            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum
    
class CrossEntropyloss_masked(nn.Module):
    def __init__(self):
        super(CrossEntropyloss_masked, self).__init__()

    def forward(self,  pred_perm, gt_perm, pred_num):

        pred_perm = pred_perm.to(dtype=torch.float32)

        try:
            assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_perm)
            raise err
        
        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)

        loss += F.binary_cross_entropy(
            pred_perm[0],
            gt_perm[0],
            reduction = 'sum'
        )
        n_sum += pred_num.to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum

