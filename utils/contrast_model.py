# select k negative samples from a big batch
# reference from GCL.models.contrast_model.py
import torch
from GCL.losses import Loss
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) 
        loss = torch.diag(sim) - torch.log(exp_sim.sum(dim=1))
        
        return -loss.mean()

class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, **kwargs):
        super().__init__()
        self.loss = loss
        self.mode = mode
        self.kwargs = kwargs
        
    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None, negative_samples=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            if negative_samples is not None:
                h1 = pad_sequence(list(torch.split(h1, negative_samples, dim=0)), batch_first=True)
                h2 = pad_sequence(list(torch.split(h2, negative_samples, dim=0)), batch_first=True)
            else:
                h1 = h1.unsuqeeze(0)
                h2 = h2.unsuqeeze(0)
            
            batch_num = h1.shape[0] if len(h1.shape) == 3 else 1
            
            loss = 0
            for i in range(batch_num):
                loss += self.loss(anchor=h1[i], sample=h2[i])
            loss = loss / batch_num
        return loss 
            
            