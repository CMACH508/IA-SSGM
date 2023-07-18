import torch
from torch_geometric.utils import to_dense_adj

# def matching_accuracy(pmat_pred, pmat_gt, ns):
#     """
#     Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
#     Matching Accuracy is equivalent to the recall of matching.
#     :param pmat_pred: predicted permutation matrix
#     :param pmat_gt: ground truth permutation matrix
#     :param ns: number of exact pairs
#     :return: matching accuracy, mean matching precision, matched num of pairs, total num of pairs
#     """
#     device = pmat_pred.device
#     batch_num = pmat_pred.shape[0]

#     pmat_gt = pmat_gt.to(device)

#     assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can noly contain 0/1 elements.'
#     assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should noly contain 0/1 elements.'
#     assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
#     assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

#     #indices_pred = torch.argmax(pmat_pred, dim=-1)
#     #indices_gt = torch.argmax(pmat_gt, dim=-1)

#     #matched = (indices_gt == indices_pred).type(pmat_pred.dtype)
#     match_num = 0
#     total_num = 0
#     acc = torch.zeros(batch_num, device = device)
#     precision = torch.zeros(batch_num, device = device)

#     for b in range(batch_num):
#         #match_num += torch.sum(matched[b, :ns[b]])
#         #total_num += ns[b].item()
#         acc[b] = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / torch.sum(pmat_gt[b, :ns[b]])
#         precision[b] = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / torch.sum(pmat_pred[b, :ns[b]])
#         match_num += torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]])
#         total_num += torch.sum(pmat_gt[b, :ns[b]])

#     acc[torch.isnan(acc)] = 1
#     precision[torch.isnan(precision)] = 1

#     return acc, precision, total_num

def matching_accuracy(pmat_pred, pmat_gt, ns):
    """
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
    Matching Accuracy is equivalent to the recall of matching.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: mean matching accuracy, matched num of pairs, total num of pairs
    """
    device = pmat_pred.device

    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can only contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should only contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    assert pmat_pred.shape == pmat_gt.shape, print(pmat_pred.shape, pmat_gt.shape)
    #indices_pred = torch.argmax(pmat_pred, dim=-1)
    #indices_gt = torch.argmax(pmat_gt, dim=-1)

    #matched = (indices_gt == indices_pred).type(pmat_pred.dtype)
    match_num = 0
    total_num = 0
    acc = torch.zeros(batch_num, device=device)
    for b in range(batch_num):
        #match_num += torch.sum(matched[b, :ns[b]])
        #total_num += ns[b].item()
        acc[b] = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / torch.sum(pmat_gt[b, :ns[b]])
        match_num += torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]])
        total_num += torch.sum(pmat_gt[b, :ns[b]])

    acc[torch.isnan(acc)] = 1

    #return match_num / total_num, match_num, total_num
    return acc, match_num, total_num


def matching_precision(pmat_pred, pmat_gt, ns):
    """
    Matching Precision between predicted permutation matrix and ground truth permutation matrix.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: mean matching precision, matched num of pairs, total num of pairs
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can only contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should only contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    match_num = 0
    total_num = 0
    precision = torch.zeros(batch_num, device=device)
    for b in range(batch_num):
        precision[b] = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / torch.sum(pmat_pred[b, :ns[b]])
        match_num += torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]])
        total_num += torch.sum(pmat_pred[b, :ns[b]])

    precision[torch.isnan(precision)] = 1

    # return match_num / total_num, match_num, total_num
    return precision, match_num, total_num


def format_accuracy_metric(ps, rs, f1s):
    """
    Helper function for formatting precision, recall and f1 score metric
    :param ps: tensor containing precisions
    :param rs: tensor containing recalls
    :param f1s: tensor containing f1 scores
    :return: a formatted string with mean and variance of precision, recall and f1 score
    """
    return 'p = {:.4f}v{:.4f}, r = {:.4f}v{:.4f}, f1 = {:.4f}v{:.4f}' \
        .format(torch.mean(ps), torch.std(ps), torch.mean(rs), torch.std(rs), torch.mean(f1s), torch.std(f1s))

def format_metric(ms):
    """
    Helping function for formatting single metric
    :param ms: tensor containing metric
    :return: a formatted string containing mean and variance
    """
    return '{:.4f}+-{:.4f}'.format(torch.mean(ms), torch.std(ms))

def objective_score(pmat_pred, affmtx, ns):
    """
    Objective score given predicted permutation matrix and affinity matrix from the problem.
    :param pmat_pred: predicted permutation matrix
    :param affmtx: affinity matrix from the problem
    :param ns: number of exact pairs
    :return: objective scores
    """
    batch_num = pmat_pred.shape[0]

    p_vec = pmat_pred.transpose(1, 2).contiguous().view(batch_num, -1, 1)
    obj_score = torch.matmul(torch.matmul(p_vec.transpose(1, 2), affmtx), p_vec).view(-1)

    return obj_score

def matching_accuarcy_mask(pmat_pred, test_y):
    """
    Matching Accuracy between full predicted permutation matrix and masked ground truth (test samples ground truth).
    :param pmat_pred: predicted permutation matrix within all nodes.
    :param test_y: [2, num_test_nodes], ground truth sparse matrix.
    :return: mean matching accuracy.
    """
    device = pmat_pred.device
    
    assert len(test_y.shape) == 2, 'test_y.shape is not [2, num_test_nodes].'

    if len(pmat_pred.shape) == 3:
        pmat_pred = pmat_pred.squeeze(0)

    pred = pmat_pred[test_y[0]].argmax(dim=-1).to(device)
    acc = (pred == test_y[1]).sum() / torch.tensor(test_y.size(1)).to(device)

    return acc

def matching_accuarcy_sparse(pmat_pred, test_y):
    """
    :param: pmat_pred: torch.sparse_coo_tensor([2, min(num_src_nodes, num_tgt_nodes)], size = (num_src_nodes, num_tgt_nodes))
    :param: test_y: [2, num_test_nodes], ground truth sparse matrix.
    """
    device = pmat_pred.device
    print(pmat_pred._indices().shape, test_y.shape )
    pred = pmat_pred._indices()[1, test_y[0]].to(device)
    acc = (pred == test_y[1]).sum() / torch.tensor(test_y.size(1)).to(device)
    
    return acc
    


def matching_hits_at_k(k, softmax_pred, test_y, reduction='mean'):
    """
    hit@k accuracy between softmax_pred matrix and ground truth. 
    :param k: int, The :math:`\mathrm{top}_k` predictions to consider.
    :param softmax_pred: tensor, dense correspondence matrix of the shapw [batch_size, full_num_nodes, full_num_nodes].
    :param test_y: tensor, [2, num_test_nodes], ground truth sparse matrix.
    :return: mean matching hit@k accuracy
    """
    assert reduction in ['mean', 'sum']

    device = softmax_pred.device
    
    if len(softmax_pred.shape) == 3:
        softmax_pred = softmax_pred.squeeze(0)
    
    pred = softmax_pred[test_y[0]].argsort(dim=-1, descending=True)[:, :k]
    acc = (pred == test_y[1].view(-1, 1)).sum() / torch.tensor(test_y.size(1))

    return acc
    

    

