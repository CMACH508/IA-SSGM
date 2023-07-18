import torch
from torch.nn import DataParallel
from collections import OrderedDict


def save_model(model, path):
    if isinstance(model, DataParallel): # 使用nn.DataParallel后，事实上DataParallel也是一个Pytorch的nn.Module，那么你的模型和优化器都需要使用.module来得到实际的模型和优化器
        model = model.module

    torch.save(model.state_dict(), path)


def load_model(model, path, strict=True):
    if isinstance(model, DataParallel):
        module = model.module
    else:
        module = model
    
    params = torch.load(path)

    missing_keys, unexpected_keys = module.load_state_dict(params, strict=strict)
    
    # cmpnn TODO:
    # params = torch.load(path)
    # missing_keys, unexpected_keys = module.load_state_dict(params['model_state_dict'], strict=strict)

    if len(unexpected_keys) > 0:
        print('Warning: Unexpected key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        print('Warning: Missing key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in missing_keys)))

