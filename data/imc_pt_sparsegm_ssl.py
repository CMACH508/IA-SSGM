# reference pytorch_geometric.datasets.
import os
import os.path as osp
import shutil
import glob
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import (Data, InMemoryDataset)

from data.full_connected import FullConnected

class IMCPTSparseGMObject(InMemoryDataset):
    r"""The IMC 
    Args:
        root (string): Root directory where the dataset should be 
            saved. 
        category (string): The category of the images.
        transform (callable, optional): A function/transform that takes in 
            an :obj:`torch_geometric.data.Data` object and returns a transformed 
            version. The data object will be transformed before every access.
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    categories = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior', 
                  'grand_place_brussels', 'hagia_sophia_interior', 'notre_dame_front_facade', 
                  'palace_of_westminster', 'pantheon_exterior', 'prague_old_town_square',
                  'taj_mahal', 'temple_nara_japan', 'trevi_fountain', 'westminster_abbey',
                  'reichstag', 'sacre_coeur', 'st_peters_square']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16

    def __init__(self, root, category, transform=None, pre_transform=None, 
                pre_filter=None):
        assert category.lower() in self.categories
        self.category = category
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')
    
    @property
    def raw_file_names(self):
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.""" 
        return [category for category in self.categories]

    @property
    def processed_dir(self):
        return osp.join(self.root, self.category, 'processed')

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return 'data.pt'
    
    def process(self):
        from PIL import Image
        from scipy.io import loadmat
        import torchvision.transforms as T
        import torchvision.models as models
        category = self.category

        image_path = osp.join(self.raw_dir, 'images')
        annotation_path = osp.join(self.raw_dir, 'annotations')
        
        names = np.load(osp.join(annotation_path, category, 'img_info.npz'))['img_name'].tolist()
        names = [name[:-4] for name in names]
        
        vgg16_outputs = []
        
        def hook(module, x, y):
            vgg16_outputs.append(y.to('cpu'))

        vgg16 = models.vgg16(pretrained=True).to(self.device)
        vgg16.eval()
        vgg16.features[20].register_forward_hook(hook)  # relu4_2
        vgg16.features[25].register_forward_hook(hook)  # relu5_1

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        data_list = []
        
        for name in names:
            keypoints = np.load(osp.join(annotation_path, category, f'{name}.npz'))['points'] # 3 X N.
            kpts_order = [i for i in range(keypoints.shape[1])]
            random.shuffle(kpts_order)
            ys = keypoints[0, :][kpts_order].astype(np.int16)
            poss = keypoints[1:, :][:, kpts_order].T

            y = torch.from_numpy(ys).long()
            pos = torch.from_numpy(poss).float().view(-1, 2)

            if pos.numel() == 0: # 获取tensor中一共有多少元素
                continue  # These examples do not make any sense anyway...
            
            path = osp.join(image_path, category, f'{name}.jpg')
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            
            # Rescale keypoints.
            pos[:, 0] = pos[:, 0] * 256.0 / (img.size[0])
            pos[:, 1] = pos[:, 1] * 256.0 / (img.size[1])

            img = img.resize((256, 256), resample=Image.BICUBIC)
            img = transform(img)

            data = Data(img=img, pos=pos, y=y, name=name)
            data_list.append(data)
        
        imgs = [data.img for data in data_list]
        loader = DataLoader(imgs, self.batch_size, shuffle=False)
        for i, batch_img in enumerate(loader):
            vgg16_outputs.clear()

            with torch.no_grad():
                vgg16(batch_img.to(self.device))

            out1 = F.interpolate(vgg16_outputs[0], (256, 256), mode='bilinear',
                                 align_corners=False)
            out2 = F.interpolate(vgg16_outputs[1], (256, 256), mode='bilinear',
                                 align_corners=False)

            for j in range(out1.size(0)):
                data = data_list[i * self.batch_size + j]
                idx = data.pos.round().long().clamp(0, 255)
                x_1 = out1[j, :, idx[:, 1], idx[:, 0]].to('cpu')
                x_2 = out2[j, :, idx[:, 1], idx[:, 0]].to('cpu')
                # data.img = None
                data.x = torch.cat([x_1.t(), x_2.t()], dim=-1)
            del out1
            del out2
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'category={self.category})')    

if __name__ == '__main__':
    import torch_geometric.transforms as T
    pre_filter = lambda data: data.pos.size(0) > 1
    pre_transform = T.Compose([
        FullConnected(),
        T.Cartesian(),
    ])
    path = osp.join('..', 'data', 'IMC_PT_SparseGM_SSL')
    datasets = [IMCPTSparseGMObject(path, cat, pre_transform=pre_transform) for cat in IMCPTSparseGMObject.categories]
