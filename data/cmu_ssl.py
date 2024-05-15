from itertools import chain
import os
import os.path as osp
import shutil
import glob
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric import transforms
from torch_geometric.data import (Data, InMemoryDataset, download_url, extract_zip)

class CMUHouseHotel(InMemoryDataset):
    categories = ['house', 'hotel']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16

    def __init__(self, root, category, transform=None, pre_transform=None, pre_filter=None):
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
        images_filenames = glob.glob(osp.join(self.raw_dir, category, 'images', '*.png'))
        data_list = []
        for image_filename in images_filenames:
            keypoints_filename = self.category + str(int(image_filename.split('.')[1][3:])+1).rjust(3, '0')
            
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
            poss = np.loadtxt(osp.join(self.raw_dir, category, category + 's', keypoints_filename), dtype=np.float32)
            keypoints = []
            for i, pos in enumerate(poss):
                dict = {'name': i, 'pos': pos}
                keypoints.append(dict)
            
            random.shuffle(keypoints)
            poss, ys = [], []
            for keypoint in keypoints:
                ys.append(keypoint['name'])
                x = float(keypoint['pos'][0])
                y = float(keypoint['pos'][1])
                poss += [x, y]

            y = torch.tensor(ys, dtype=torch.long)
            pos = torch.tensor(poss, dtype=torch.float).view(-1, 2)

            if pos.size(0) != 30:
                continue

            with open(str(image_filename), 'rb') as f:
                img = Image.open(f)             
                if not img.mode == 'RGB':
                    img = img.convert('RGB')
            
            # Rescale keypoints.
            pos[:, 0] = pos[:, 0] * 256.0 / (img.size[0])
            pos[:, 1] = pos[:, 1] * 256.0 / (img.size[1])
            
            img = img.resize((256, 256), resample=Image.BICUBIC)
            img = transform(img)

            data = Data(img=img, pos=pos, name=image_filename, y=y)
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
    pre_transform = T.Compose([
        T.Delaunay(),
        T.FaceToEdge(),
        T.Cartesian(),
    ])
    path = osp.join('Cmu_hotel_house_SSL')
    datasets = [CMUHouseHotel(path, cat, pre_transform) for cat in CMUHouseHotel.categories]
 
    