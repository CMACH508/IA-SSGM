import numpy as np
from easydict import EasyDict as edict
import yaml
import os

# 创建dict
__C = edict()
cfg = __C

# Minibatch size
__C.BATCH_SIZE = 4

# path to load pretrained model weights
__C.PRETRAINED_PATH = ''

# CMU-hotel-house Dataset
__C.CMU = edict()
__C.CMU.ROOT_DIR = 'data/Cmu-hotel-house'
__C.CMU.CLASSES = ['house', 'hotel']
__C.CMU.KPT_LEN = 30
__C.CMU.TRAIN_NUM = 5
__C.CMU.TRAIN_OFFSET = 0
__C.CMU.NS_SRC = 30
__C.CMU.NS_TGT = 30

# pascal cars-motorbikes
__C.PAC = edict()
__C.PAC.ROOT_DIR = 'data/PAC'
__C.PAC.CLASSES = ['Carss', 'Motor']
__C.PAC.CLASSES_FEA = ['Carss_scf','Motor_scf']
__C.PAC.KPT_LEN = 40
__C.PAC.TRAIN_NUM = 5
__C.PAC.TRAIN_OFFSET = 0

# Willow-Object Dataset
__C.WILLOW = edict()
__C.WILLOW.ROOT_DIR = 'data/WILLOW-ObjectClass'
__C.WILLOW.CLASSES = ['Car', 'Duck', 'Face', 'Motorbike', 'Winebottle']
__C.WILLOW.KPT_LEN = 10
__C.WILLOW.TRAIN_NUM = 20
__C.WILLOW.TRAIN_OFFSET = 0
__C.WILLOW.RAND_OUTLIER = 0

# VOC2011-Keypoint Dataset
__C.VOC2011 = edict()
__C.VOC2011.KPT_ANNO_DIR = 'data/PascalVOC/annotations/'  # keypoint annotation
__C.VOC2011.ROOT_DIR = 'data/PascalVOC/VOC2011/'  # original VOC2011 dataset
__C.VOC2011.SET_SPLIT = 'data/PascalVOC/voc2011_pairs.npz'  # set split path
__C.VOC2011.CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                       'tvmonitor']

# CUB2011 dataset
__C.CUB2011 = edict()
__C.CUB2011.ROOT_PATH = 'data/CUB_200_2011'
__C.CUB2011.CLASS_SPLIT = 'ori' # choose from 'ori' (original split), 'sup' (super class) or 'all' (all birds as one class)

# IMC_PT_SparseGM dataset
__C.IMC_PT_SparseGM = edict()
__C.IMC_PT_SparseGM.CLASSES = {'train': ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior',
                                      'grand_place_brussels', 'hagia_sophia_interior', 'notre_dame_front_facade',
                                      'palace_of_westminster', 'pantheon_exterior', 'prague_old_town_square',
                                      'taj_mahal', 'temple_nara_japan', 'trevi_fountain', 'westminster_abbey'],
                            'test': ['reichstag', 'sacre_coeur', 'st_peters_square']}
__C.IMC_PT_SparseGM.ROOT_DIR_NPZ = 'data/IMC_PT_SparseGM/annotations/'
__C.IMC_PT_SparseGM.ROOT_DIR_IMG = 'data/IMC_PT_SparseGM/images'
__C.IMC_PT_SparseGM.TOTAL_KPT_NUM = 50


# Pairwise data loader settings.
__C.PAIR = edict()
__C.PAIR.RESCALE = (256, 256)  # rescaled image size
__C.PAIR.GT_GRAPH_CONSTRUCT = 'tri'
__C.PAIR.REF_GRAPH_CONSTRUCT = 'fc'

#
# Problem settings. Set these parameters the same for fair comparison.
#
__C.PROBLEM = edict()

# Rescaled image size
__C.PROBLEM.RESCALE = (256, 256)
# Allow outlier in source graph. Useful for 2GM
__C.PROBLEM.SRC_OUTLIER = False
# Allow outlier in target graph. Useful for 2GM
__C.PROBLEM.TGT_OUTLIER = False

#
# Graph construction settings.
#
__C.GRAPH = edict()

# The ways of constructing source graph/target graph.
# Candidates can be 'tri' (Delaunay triangulation), 'fc' (Fully-connected)
__C.GRAPH.SRC_GRAPH_CONSTRUCT = 'tri'
__C.GRAPH.TGT_GRAPH_CONSTRUCT = 'fc'

# Build a symmetric adjacency matrix, else only the upper right triangle of adjacency matrix will be filled
__C.GRAPH.SYM_ADJACENCY = True

# Padding length on number of keypoints for batched operation
__C.GRAPH.PADDING = 23


# GMN model options
__C.GMN = edict()
__C.GMN.FEATURE_CHANNEL = 512
__C.GMN.PI_ITER_NUM = 50
__C.GMN.PI_STOP_THRESH = 2e-7
__C.GMN.BS_ITER_NUM = 10
__C.GMN.BS_EPSILON = 1e-10
__C.GMN.VOTING_ALPHA = 2e8
__C.GMN.L2_NORM_K = 0.0
__C.GMN.GNN_LAYER_NUM = 1

# BIIA model options
__C.BIIA = edict()
__C.BIIA.FEATURE_CHANNEL = 60
__C.BIIA.BS_ITER_NUM = 20
__C.BIIA.BS_EPSILON = 1.0e-10
__C.BIIA.VOTING_ALPHA = 20.
__C.BIIA.GNN_LAYER = 5
__C.BIIA.GNN_FEAT = 1024
__C.BIIA.ITERATION_ = 4
__C.BIIA.FEATURE_EXTRACTION = 'vgg16_feature'
__C.BIIA.SHAPE_CONTEXT_THETA = 12
__C.BIIA.SHAPE_CONTEXT_RADIUS = 5
__C.BIIA.ALPHA1 = 0.75
__C.BIIA.ALPHA2 = 1.25

# GCL model options
__C.GCL = edict()
__C.GCL.ENCODER_TYPE = 'GATConv'

# BBGM model options
__C.BBGM = edict()
__C.BBGM.FEATURE_CHANNEL = 512
__C.BBGM.PI_ITER_NUM = 25
__C.BBGM.BS_ITER_NUM = 20
__C.BBGM.BS_EPSILON = 1.0e-10

# CMPNN model options
__C.CMPNN = edict()
__C.CMPNN.BS_ITER_NUM = 20
__C.CMPNN.BS_EPSILON = 1.0e-10
__C.CMPNN.ITERATION_ = 4
#
# Training options
#

__C.TRAIN = edict()

# Iterations per epochs
__C.TRAIN.EPOCH_ITERS = 7000

# Training start epoch. If not 0, will be resumed from checkpoint.
__C.TRAIN.START_EPOCH = 0

# Total epochs
__C.TRAIN.NUM_EPOCHS = 30

# Optimizer type
__C.TRAIN.OPTIMIZER = 'SGD'

# Start learning rate
__C.TRAIN.LR = 0.01

# Use separate learning rate for the CNN backbone
__C.TRAIN.SEPARATE_BACKBONE_LR = False

# Start learning rate for backbone
__C.TRAIN.BACKBONE_LR = __C.TRAIN.LR

# Learning rate decay
__C.TRAIN.LR_DECAY = 0.1

# Learning rate decay step (in epochs)
__C.TRAIN.LR_STEP = [10, 20]

# SGD momentum
__C.TRAIN.MOMENTUM = 0.9

# RobustLoss normalization
__C.TRAIN.RLOSS_NORM = max(__C.PROBLEM.RESCALE)

# Specify a class for training
__C.TRAIN.CLASS = 'none'

# Loss function. Should be 'offset' or 'perm'
__C.TRAIN.LOSS_FUNC = 'perm'

# SSL Loss function mode. Should be 'G2G' or 'L2L'
__C.TRAIN.LOSS_FUNC_MODE = 'G2G'

# Train Process. Should be 'pre_train' or 'fine_tune
__C.TRAIN.PROCESS = 'pre_train'

__C.TRAIN.LOSS_GCL = 1.0

__C.TRAIN.LOSS_PERM = 1.0

__C.TRAIN.LOSS_CONSISTENCY = 1.0
#
# Evaluation options
#

__C.EVAL = edict()

# Evaluation epoch number
__C.EVAL.EPOCH = 30
__C.EVAL.EPOCH_ITERS = 100

# PCK metric
__C.EVAL.PCK_ALPHAS = []
__C.EVAL.PCK_L = float(max(__C.PROBLEM.RESCALE))  # PCK reference.

# Number of samples for testing. Stands for number of image pairs in each classes (VOC)
__C.EVAL.SAMPLES = 1000

# Evaluated classes
__C.EVAL.CLASS = 'all'

#
# MISC
#

# name of backbone net
__C.BACKBONE = 'VGG16_bn'

# Parallel GPU indices ([0] for single GPU)
__C.GPUS = [0]

# num of dataloader processes
__C.DATALOADER_NUM = __C.BATCH_SIZE

# Mean and std to normalize images
__C.NORM_MEANS = [0.485, 0.456, 0.406]
__C.NORM_STD = [0.229, 0.224, 0.225]

# Data cache path
__C.CACHE_PATH = 'data/cache'

# Model name and dataset name
__C.MODEL_NAME = ''
__C.DATASET_NAME = ''
__C.DATASET_FULL_NAME = 'PascalVOC' # 'PascalVOC' or 'WillowObject'
__C.DATASET_PATH = './data/PascalVOC_SSL'

# Module path of module
__C.MODULE = ''

# Output path (for checkpoints, running logs and visualization results)
__C.OUTPUT_PATH = ''

# The step of iteration to print running statistics.
# The real step value will be the least common multiple of this value and batch_size
__C.STATISTIC_STEP = 100

# random seed used for data loading
__C.RANDOM_SEED = 123



def lcm(x, y):
    """
    Compute the least common multiple of x and y. This function is used for running statistics.
    """
    greater = max(x, y)
    while True:
        if (greater % x == 0) and (greater % y == 0):
            lcm = greater
            break
        greater += 1
    return lcm


def get_output_dir(model, dataset):
    """
    Return the directory where experimental artifacts are placed.
    :param model: model name
    :param dataset: dataset name
    :return: output path (checkpoint and log), visual path (visualization images)
    """
    outp_path = os.path.join('output', '{}_{}'.format(model, dataset))
    return outp_path

# 内部方法，实现yaml配置文件到dict的合并
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v
# 自动加载yaml文件
def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r', encoding='utf-8') as f:
        yaml_cfg = edict(yaml.load(f,Loader= yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
