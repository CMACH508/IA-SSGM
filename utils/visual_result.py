# 想写一个程序，可视化，匹配结果
# reference COTR

import os
import cv2
import torch
import imageio
from datetime import datetime
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from os.path import basename as basename

from utils.config import cfg

def show_results(Images, Ps, pre_perm_mat: torch.Tensor, gt_perm_mat: torch.Tensor, recall: str, Fns=None):
    '''Param: Images: plt.Iamge obj
              Ps: Keypoints coordinate.
              Fns: image_filename.
              pre_perm_mat: predicted matching results.
              gt_perm_mat: ground truth matching results.
    '''
    P1, P2 = Ps
    
    image1, image2 = Images # batch ,channel, h, w
    
    P1 = P1.detach().cpu()
    P2 = P2.detach().cpu()

    if Fns is not None:
        image1_path, image2_path = Fns

    batch_size = P1.shape[0]
    for b in range(batch_size):
        # Tensor转成PIL.Image重新显示
        img_src = transforms.ToPILImage()(image1[b]).convert('RGB') #  size:(w,h)
        img_tgt = transforms.ToPILImage()(image2[b]).convert('RGB')

        img_src = np.asarray(img_src) # (h,w,c)
        img_tgt = np.asarray(img_tgt)

        # 把两张图上下拼接起来
        h1, w1 = img_src.shape[:2]
        h2, w2 = img_tgt.shape[:2]
        
        img = np.zeros((h1 + h2, max(w1, w2), 3), dtype=img_src.dtype)
        img[:h1, :w1] = img_src
        img[h1:, :w2] = img_tgt

        # recompute the coordinates for the second image
        # P2 的高度 + h1
        _P2 = P2[b] + torch.Tensor([[0, h1]]) 
        # P2 的顺序变为和P1的点相匹配的顺序
        assert _P2.shape[-2] == pre_perm_mat.shape[-1], "error num of keypoints."
        assert _P2.shape[-2] == gt_perm_mat.shape[-1], "error num of keypoints."

        pre_perm_P2 = torch.mm(pre_perm_mat[b], _P2.cuda()).detach().cpu().numpy()
        gt_perm_P2 = torch.mm(gt_perm_mat[b], _P2.cuda()).detach().cpu().numpy()

        fig = plt.figure(frameon=False) # 不显示边框
        fig = plt.imshow(img.astype(np.uint8)) # heat map

        # 绘图参数
        cols = [
            [0.0, 0.67, 0.0],
            [0.9, 0.1, 0.1],
        ]
        lw = .2
        alpha = 1

        # source 和 target 图中的所有的 x, y 坐标，stack 起来
        pre_Xs = np.stack([P1[b][:, 0], pre_perm_P2[:, 0]], axis= 1).T
        pre_Ys = np.stack([P1[b][:, 1], pre_perm_P2[:, 1]], axis= 1).T
        
        gt_Xs = np.stack([P1[b][:, 0], gt_perm_P2[:, 0]], axis= 1).T
        gt_Ys = np.stack([P1[b][:, 1], gt_perm_P2[:, 1]], axis= 1).T

        P_Xs = np.concatenate((P1[b][:, 0], _P2[:, 0]), axis=0).T
        P_Ys = np.concatenate((P1[b][:, 1], _P2[:, 1]), axis=0).T

        # 如果 X 和 Y 均为矩阵，则它们的大小必须相同。plot 函数绘制 Y 的列对 X 的列的图。
        plt.plot(
            pre_Xs, pre_Ys, 
            alpha=alpha,  # alpha 透明度
            linestyle="-",
            linewidth=lw,
            aa=False,
            color=cols[0],
        )

        # plt.scatter(pre_Xs, pre_Ys) # 散点图
        plt.scatter(P_Xs, P_Ys) # 散点图

        plt.plot(
            gt_Xs, gt_Ys,
            alpha=alpha, 
            linestyle=":",
            linewidth=lw,
            aa=False,
            color=cols[1],
        )       

        # 关闭子图的轴
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        ax = plt.gca()
        ax.set_axis_off()
        # plt.show()
        if Fns is not None:
            imgname_src = basename(image1_path[b]).split('.')[0]
            imgname_tgt = basename(image2_path[b]).split('.')[0]
            # imgname_src = image1_path[b]
            # imgname_tgt = image2_path[b]
            plt_path = cfg.OUTPUT_PATH + '/matching_visible/' + recall + '/'
            if not Path(plt_path).exists():
                Path(plt_path).mkdir(parents=True)
            plt.savefig(plt_path + imgname_src + 'VS' + imgname_tgt + '.png')
        else:
            plt_path = cfg.OUTPUT_PATH + '/matching_visible/' + recall + '/'
            if not Path(plt_path).exists():
                Path(plt_path).mkdir(parents=True)
            now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            plt.savefig(plt_path + now_time + '.png')
        plt.close()


if __name__ == "__main__":
    image1_path = ['/home/guowenqi/GraphMatching/Unsup-GM/data/PascalVOC_SSL/raw/images/JPEGImages/2007_000039.jpg']
    image2_path = ['/home/guowenqi/GraphMatching/Unsup-GM/data/PascalVOC_SSL/raw/images/JPEGImages/2007_000187.jpg']
    
    img_src = Image.open(str(image1_path[0])) 
    img_tgt = Image.open(str(image2_path[0]))  
    
    images = [img_src, img_tgt]
    image_filenames = [image1_path, image2_path]
    P1 = torch.Tensor([[[333.54, 106.51],[161.95,96.91], [326.48,267.42],[166.89,237.49]]])
    P2 = torch.Tensor([[[236.49, 106.22],[5.51,137.09], [238.50,279.38],[21.83,327.07]]])
    Ps = [P1, P2]
    perm_mat = torch.Tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]])
    show_results(images, Ps, image_filenames, perm_mat)
    
