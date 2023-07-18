# 重新实现非矩阵形式的 decoupled infoNCE loss function.
# 目的是为了大的batch_size 不会爆显存.
# 具体公式：
# L_{D C}=\sum_{k \in\{1,2\}, i \in \llbracket 1, N \rrbracket} L_{D C, i}^{(k)}
# L_{D C,i} ^ (k) = -\left\langle\mathbf{z}_{i}^{(1)}, \mathbf{z}_{i}^{(2)}\right\rangle / \tau+\log \sum_{l \in\{1,2\}, j \in \llbracket 1, N \rrbracket, j \neq i} \exp \left(\left\langle\mathbf{z}_{i}^{(k)}, \mathbf{z}_{j}^{(l)}\right\rangle / \tau\right)
# 
# input: anchor: 一个batch中所有的 node/graph 个数和 * feature_channel, 即 N * feature_channel.
#        sample: the shape same as the anchor.
# process:  loss_sum = 0
        # for i in range(N):
        #     sim = similarity(anchor[i], sample) / tau.
        #     pos_loss = - sim[i]
        #     neg_loss = torch.logsumexp(sim[torch.arrange(N)!=i])
        #     loss = pos_loss + neg_loss
        #     loss_sum += loss
# return: loss_sum.mean()


# for G2L:
# process: loss_sum = 0
        # for i in range(num_graph):
        #     sim = _similarity(anchor[i], sample)  # 1 X N
        #     sim = torch.exp(sim / tau)

        #     neg_mask[i] = 1 = pos_mask[i]
        #     pos = (sim * pos_mask[i]).sum(dim = 1)
        #     neg = (sim * neg_mask[i]).sum(dim = 1)

        #     loss = pos / (pos + neg)
        #     loss = -torch.log(loss)
        #     loss_sum += loss
# return loss.mean()


import torch

import GCL.losses.losses as L
import torch.nn.functional as F

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1,dim = 0)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class DecoupledInfoNCE(L.Loss):
    def __init__(self, tau):
        super(DecoupledInfoNCE, self).__init__()
        self.tau = tau
    
    def compute(self, anchor, sample, pos_mask = None, neg_mask = None, *args, **kwargs):
        loss_sum = 0
        for i in range(anchor.shape[0]):
            # sim = _similarity(anchor[i], sample)
            sim = torch.cosine_similarity(anchor[i].unsqueeze(0), sample, dim=1)
            sim = torch.exp(sim / self.tau)

            neg_mask = 1 - pos_mask[i]
            pos_loss = (sim * pos_mask[i]).sum(dim = 0)
            neg_loss = (sim * neg_mask).sum(dim = 0)
            loss = - torch.log(pos_loss / (pos_loss + neg_loss))
            loss_sum += loss            
        
        return loss_sum / (anchor.shape[0])

class INfoNCE(L.Loss):
    def __init__(self, tau):
        super(INfoNCE, self).__init__()
        self.tau = tau
    
    def compute(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs):
        loss_sum = 0
        for i in range(anchor.shape[0]):
            pos_loss, neg_loss = 0, 0
            for j in range(sample.shape[0]):
                h1 = F.normalize(anchor[i], dim=0)
                h2 = F.normalize(sample[j], dim=0)
                if i == j:
                    pos_loss += torch.exp((h1 @ h2) / self.tau)
                else:
                    neg_loss += torch.exp((h1 @ h2) / self.tau)
            loss = - torch.log(pos_loss / (pos_loss + neg_loss))
            loss_sum += loss

        return loss_sum / (anchor.shape[0])
 
if __name__ == '__main__':
    anchor = torch.randn([3,5])
    sample = torch.randn([4,5])
    pos_mask = torch.eye(4, dtype=torch.float32)
    neg_mask = 1. - pos_mask
    loss_fuc1 = INfoNCE(tau=0.2)
    loss_fuc2 = DecoupledInfoNCE(tau=0.2)
    loss1, pos1, neg1 = loss_fuc1.compute(anchor, sample)
    loss2, pos2, neg2  = loss_fuc2.compute(anchor, sample, pos_mask, neg_mask)
    print(loss1 == loss2, loss1, loss2, pos1, pos2, neg1, neg2)