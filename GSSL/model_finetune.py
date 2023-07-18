# fine_tune model_pretrain & Matching

import torch
import torch.nn as nn
from torch.utils import data

from GSSL.model_pretrain import GCL
from GSSL.Matching import Matching
from utils.model_sl import load_model
from utils.config import cfg

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # load pre_train GCL model.      
        self.encoder = GCL()
        if len(cfg.PRETRAINED_PATH) > 0:
            encoder_model_path = cfg.PRETRAINED_PATH
            print('Loading model from {}'.format(encoder_model_path))
            load_model(self.encoder, encoder_model_path, strict=False)
        if cfg.TRAIN.BACKBONE_LR == 0.0:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.matching = Matching()
    
    def forward(self, data_dict):
        self.encoder(data_dict)
        self.matching(data_dict)

        return data_dict

        
        