# show the results of two graph and graph matching result.
## different nodes have different colors.
## hope the line is a curve.
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

import torch
from torchvision import transforms
from torch_geometric.utils import to_dense_batch



def visible_compare_pair_graph(images, 
                               Ps,
                               Ys,
                               edge_indexs, 
                               batchs, 
                               device, 
                               filenames=None, 
                               pred_permutation_ms=None, 
                               ground_truth_ms=None):
    # images process
    imgs_src, imgs_tgt = images

    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)

    imgs_src = imgs_src.view(-1, 3, 256, 256)
    imgs_src = (imgs_src.permute(0, 2, 3, 1) * std + mean).permute(0, 3, 1, 2)

    imgs_tgt = imgs_tgt.view(-1, 3, 256, 256)
    imgs_tgt = (imgs_tgt.permute(0, 2, 3, 1) * std + mean).permute(0, 3, 1, 2)

    Ps_src, Ps_tgt = Ps
    Ys_src, Ys_tgt = Ys
    x_s_batchs, x_t_batchs = batchs

    # Ps, Ys, to dense batch
    Ps_src, src_masks = to_dense_batch(Ps_src, x_s_batchs)
    Ps_tgt, tgt_masks = to_dense_batch(Ps_tgt, x_t_batchs)


    Ps_tgt_pre = torch.bmm(pred_permutation_ms.to(device), Ps_tgt)
    Ps_tgt_gt = torch.bmm(ground_truth_ms.to(device), Ps_tgt)

    Ys_src, src_masks = to_dense_batch(Ys_src, x_s_batchs, fill_value=0.0)
    Ys_tgt, tgt_masks = to_dense_batch(Ys_tgt, x_t_batchs, fill_value=0.0)

    if filenames is not None:
        files_src, files_tgt = filenames

    batch_size = x_s_batchs.max()
    for b in range(batch_size):
        img_src = transforms.ToPILImage()(imgs_src[b]).convert('RGB')
        img_tgt = transforms.ToPILImage()(imgs_tgt[b]).convert('RGB')

        img_src = np.asarray(img_src)
        img_tgt = np.asarray(img_tgt)
    
        # 把两张图左右拼接起来
        h_src, w_src = img_src.shape[:2]
        h_tgt, w_tgt = img_tgt.shape[:2]

        img = np.zeros((max(h_src, h_tgt), w_src + w_tgt, 3),dtype=img_src.dtype)
        img[:h_src, :w_src] = img_src
        img[:h_tgt, w_src:] = img_tgt

        # 重新计算 target image 的 keypints coordinates.
        P_tgt = Ps_tgt[b].cpu() + torch.tensor([[w_src, 0]])
        P_tgt_pre = Ps_tgt_pre[b].cpu() + torch.tensor([[w_src, 0]])
        P_tgt_gt = Ps_tgt_gt[b].cpu() + torch.tensor([[w_src, 0]])

        fig = plt.figure(frameon=False) # 不显示边框
        fig = plt.imshow(img.astype(np.uint8))
        
        P_src = Ps_src[b].cpu().numpy()
        src_mask = src_masks[b].cpu()
        P_tgt.numpy()
        P_tgt_pre.numpy() 
        P_tgt_gt.numpy()
        
        Y_src = Ys_src[b].cpu().numpy()
        Y_tgt = Ys_tgt[b].cpu().numpy()
        
        
        # 散点图
        plt.scatter(P_src[:, 0].T, P_src[:, 1].T,
                    alpha=0.8,
                    marker='o',
                    s=15,
                    # c=Y_src,
                    edgecolors= 'green',
                    color = 'lime',
                    # cmap='coolwarm'
                    )

        plt.scatter(P_tgt[:, 0].T, P_tgt[:, 1].T,
                    alpha=0.8,
                    marker='o',
                    s=15,
                    # c=Y_tgt,
                    edgecolors= 'green',
                    color = 'lime',
                    # cmap='coolwarm',
                    )
        
        # # 预测的匹配
        # for i in range(P_src[src_mask].shape[0]):
        #     plt.annotate(
        #         "",
        #         xy=P_src[i], 
        #         xytext=P_tgt_pre[i],
        #         size=20, va="center", ha="center",
        #         arrowprops=dict(color='#373331',
        #                         arrowstyle="-",
        #                         connectionstyle="arc3,rad=0.4",
        #                         linewidth=2
        #                         )

        #     )
        # 真实的匹配
        for i in range(P_src[src_mask].shape[0]):
            plt.plot(
                np.stack([P_src[:, 0], P_tgt_gt[:, 0]], axis=1).T,
                np.stack([P_src[:, 1], P_tgt_gt[:, 1]], axis=1).T,
                alpha=0.5,
                linewidth=0.5,
                color='tomato',
            )
    
        # 预测的匹配
        for i  in range(P_src[src_mask].shape[0]):
            plt.plot(
                np.stack([P_src[:, 0], P_tgt_pre[:, 0]], axis=1).T,
                np.stack([P_src[:, 1], P_tgt_pre[:, 1]], axis=1).T,
                alpha=0.5,
                linewidth=0.5,
                color='lime',
                
            )
        # 关闭子图的轴
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        ax = plt.gca()
        ax.set_axis_off()
        
        if filenames is not None:
            imgname_src = osp.basename(files_src[b]).split('.')[0]
            imgname_tgt = osp.basename(files_tgt[b]).split('.')[0]
            plt.savefig(imgname_src + 'VS' + imgname_tgt + '.png')
            plt.savefig(imgname_src + 'VS' + imgname_tgt + '.pdf')
        plt.close()
        

            

        
        
        
        



        

    
    