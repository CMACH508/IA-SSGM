

import random
import numpy as np
import os.path as osp
from PIL import Image
from itertools import chain

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.data import (InMemoryDataset, Data)

from data.full_connected import FullConnected
from data.hippi_distance import HIPPIDistance

def lists2dict(keys, vals):
        ans = {}
        for idx, val_i in enumerate(vals):
            if keys[idx] in ans:
                ans[keys[idx]].append(val_i)
            else:
                ans[keys[idx]] = [val_i]
        return ans

class CUB2011Object(InMemoryDataset):
    r"""
    
    Args:
        category (string): The category of the images (one of
            :obj:`"Aeroplane"`, :obj:`"Bicycle"`, :obj:`"Bird"`,
            :obj:`"Boat"`, :obj:`"Bottle"`, :obj:`"Bus"`, :obj:`"Car"`,
            :obj:`"Cat"`, :obj:`"Chair"`, :obj:`"Diningtable"`, :obj:`"Dog"`,
            :obj:`"Horse"`, :obj:`"Motorbike"`, :obj:`"Person"`,
            :obj:`"Pottedplant"`, :obj:`"Sheep"`, :obj:`"Sofa"`,
            :obj:`"Train"`, :obj:`"TVMonitor"`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    with open('data/CUB_200_2011_SSL/raw/classes.txt') as f:
        categories = list(l.rstrip('\n').split()[1] for l in f.readlines())
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16

    def __init__(self, root, category, train=True, transform=None, pre_transform=None, pre_filter=None):
        self.category = category
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.""" 
        return ['images', 'attributes', 'parts', 'bounding_boxes.txt', 'classes.txt', 'image_class_labels.txt', 'images.txt', 'train_test_split.txt']

    @property
    def processed_dir(self):
        return osp.join(self.root, self.category.capitalize(), 'processed') # capitalize()将字符串第一个字母大写，其他小写

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return ['training.pt', 'test.pt']


    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        from PIL import Image
        import torchvision.transforms as T
        import torchvision.models as models
        
        with open(osp.join(self.raw_dir, 'images.txt')) as f:
            img2fn = dict(l.rstrip('\n').split() for l in f.readlines()) # 删除 string 字符串末尾的指定字符（默认为空格)
        with open(osp.join(self.raw_dir, 'train_test_split.txt')) as f:
            train_test_split = dict(l.rstrip('\n').split() for l in f.readlines())
        with open(osp.join(self.raw_dir, 'classes.txt')) as f:
            classes = dict(l.rstrip('\n').split() for l in f.readlines())
        with open(osp.join(self.raw_dir, 'image_class_labels.txt')) as f:
            img2class = [l.rstrip('\n').split() for l in f.readlines()]
            img_idxs, class_idxs = map(list, zip(*img2class))
            class2img = lists2dict(class_idxs, img_idxs) # 转换成class id --> img id
        with open(osp.join(self.raw_dir, 'parts', 'part_locs.txt')) as f:
            part_locs = [l.rstrip('\n').split() for l in f.readlines()]
            fi, pi, x, y, v = map(list, zip(*part_locs)) # fi: image_id, pi: part_id, x and y: pixel location of the center of the part, v: 0 if the part is not visible in the image and 1 otherwise.
            im2kpts = lists2dict(fi, zip(pi, x, y, v))
        with open(osp.join(self.raw_dir, 'bounding_boxes.txt')) as f:
            bboxes = [l.rstrip('\n').split() for l in f.readlines()]
            ii, x, y, w, h = map(list, zip(*bboxes)) # ii: image_id
            im2bbox = dict(zip(ii, zip(x, y, w, h)))

        train_split, test_split = [], []
    
        category_idx = self.categories.index(self.category) + 1
        for img_idx in class2img[str(category_idx)]:
            if train_test_split[img_idx] == '1':
                train_split.append(img_idx)
            else:
                test_split.append(img_idx)
        

        image_path = osp.join(self.raw_dir, 'images')

        labels = {}
        
        vgg16_outputs = []
        
        def hook(module, x, y):
            vgg16_outputs.append(y)

        vgg16 = models.vgg16(pretrained=True).to(self.device)
        vgg16.eval()
        vgg16.features[20].register_forward_hook(hook) # relu4_2
        vgg16.features[25].register_forward_hook(hook) # relu5_1

        transform = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_set, test_set = [], []
        for i, img_idx in enumerate(chain(train_split, test_split)):
            # image_file
            filename = img2fn[img_idx]
            
            # keypoints box
            xmin, ymin, w, h = im2bbox[img_idx]
            box = (float(xmin), float(ymin), float(w)+ float(xmin), float(h) + float(ymin))

            # pi: part_id 相当于labels， 关键点的name
            pi, x, y, v = map(list, zip(*im2kpts[img_idx]))            
            order = np.argsort(np.array(pi).astype(int))
            keypoints = np.array([np.array(x).astype('float')[order], np.array(y).astype('float')[order]])
            visible = np.array(v).astype('uint8')[order]
            
            
            poss, ys = [], []
            kpt_order = [i for i in range(keypoints.shape[1])]
            random.shuffle(kpt_order)
            for kpt_idx in kpt_order:
                if visible[kpt_idx]:
                    ys.append(int(pi[kpt_idx]))
                    x = (keypoints[0, kpt_idx] - box[0]) * 256.0 / (box[2] - box[0])
                    y = (keypoints[1, kpt_idx] - box[1]) * 256.0 / (box[3] - box[1])
                    poss += [x, y]
                    
            y = torch.tensor(ys, dtype=torch.long)
            pos = torch.tensor(poss, dtype=torch.float).view(-1, 2)

            if pos.numel() == 0: # 获取tensor中一共有多少元素
                continue  # These examples do not make any sense anyway...
            
            path = osp.join(image_path, filename)
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB').crop(box)
                img = img.resize((256, 256), resample=Image.BICUBIC)

            img = transform(img)

            data = Data(img=img, pos=pos, y=y, name=filename)
            
            if i < len(train_split):
                train_set.append(data)
            else:
                test_set.append(data)
        
        data_list = list(chain(train_set, test_set))
        imgs = [data.img for data in data_list]
        loader = DataLoader(imgs, self.batch_size, shuffle=False)
        for i, batch_img in enumerate(loader):
            vgg16_outputs.clear()

            with torch.no_grad():
                vgg16(batch_img.to(self.device))

            out1 = F.interpolate(vgg16_outputs[0], (256, 256), mode='bilinear', align_corners=False)
            out2 = F.interpolate(vgg16_outputs[1], (256, 256), mode='bilinear', align_corners=False)

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
            train_set = [data for data in train_set if self.pre_filter(data)]
            test_set = [data for data in test_set if self.pre_filter(data)]

        if self.pre_transform is not None:
            train_set = [self.pre_transform(data) for data in train_set]
            test_set = [self.pre_transform(data) for data in test_set]
        
        torch.save(self.collate(train_set), self.processed_paths[0])
        torch.save(self.collate(test_set), self.processed_paths[1])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'category={self.category})')


# if __name__ == '__main__':
#     pre_filter = lambda data: data.pos.size(0) > 1
#     pre_transform = T.Compose([
#         T.Delaunay(),   # Computes the delaunay triangulation of a set of points.
#         T.FaceToEdge(), # Converts mesh faces [3, num_faces] to edge indices [2, num_edges].
#         T.Cartesian(),  # Saves the relative Cartesian coordinates of linked nodes in its edge attributes.
#     ])
    
#     path = osp.join('..', 'data', 'CUB_200_2011_SSL_')
#     for category in CUB2011Object.categories:
#         dataset = CUB2011Object(path, category, train=True, pre_filter=pre_filter, pre_transform=pre_transform)
#         dataset = CUB2011Object(path, category, train=False, pre_filter=pre_filter, pre_transform=pre_transform)

# create full-connected graph.
if __name__ == '__main__':
    pre_filter = lambda data: data.pos.size(0) > 1
    # pre_transform = T.Compose([
    #     FullConnected(),
    #     HIPPIDistance(),
    # ])
    pre_transform = T.Compose([
        FullConnected(),
        T.Cartesian(),
    ])

    path = osp.join('..', 'data', 'CUB_200_2011_SSL')
    for category in CUB2011Object.categories:
        dataset = CUB2011Object(path, category, train=True, pre_filter=pre_filter, pre_transform=pre_transform)
        dataset = CUB2011Object(path, category, train=False, pre_filter=pre_filter, pre_transform=pre_transform)
