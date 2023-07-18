import torch
from torch_geometric.utils import to_dense_adj

def generate_y(y, batch, s_num_node, t_num_node):
    r"""Transfer the data.y into a matrix shape. 
    Args:
        y (LongTensor): Ground-truth matchings of shape [num_ground_truths]
        batch (LongTensor): Source graph batch vector of shape [batch_size * num_nodes]
        s_num_node (LongTensor): Source graph num node of shape[batch_size]
        t_num_node (LongTensor): Target graph num node of shape[batch_size]
    """
    batch_size = batch.max().item() + 1
    s_num_node_max = s_num_node.max().item()
    t_num_node_max = t_num_node.max().item()
    ret_binary_m = torch.zeros([batch_size, s_num_node_max, t_num_node_max])

    row_index =[j for i in range(len(s_num_node)) for j in range(s_num_node[i])]
    for i in range(len(batch)):
        ret_binary_m[batch[i]][row_index[i]][y[i]] = 1
    
    return ret_binary_m

def generate_masked_y(y):
    assert len(y.shape) == 2

    gt_m = to_dense_adj(y).squeeze(0)
    gt_m = gt_m[y[0]].transpose(0, 1)
    gt_m = gt_m[y[1]].transpose(0, 1)

    return gt_m.unsqueeze(0)

    


