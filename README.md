# Self-Supervised Bidirectional Learning of Graph Matching
This repository contains PyTorch implementation of the paper: [Self-Supervised Bidirectional Learning of Graph Matching](https://).

`Abstract: Deep learning methods have demonstrated promising per- formance on the NP-hard Graph Matching (GM) problems. However, the state-of-the-art methods usually require the ground-truth labels, which may take extensive human efforts or be impractical to collect. In this paper, we present a robust self-supervised bidirectional learning method (IA-SSGM) to tackle GM in an unsupervised manner. It involves an affinity learning component and a classic GM solver. Specifically, we adopt the Hungarian solver to generate pseudo correspon- dence labels for the simple probabilistic relaxation of the affinity matrix. In addition, a bidirectional recycling consistency module is proposed to generate pseudo samples by recycling the pseudo correspondence back to permute the input. It imposes a consistency constraint between the pseudo affinity and the original one, which is theoretically supported to help reduce the matching error. Our method further develops a graph contrastive learning jointly with the affinity learning to enhance its robustness against the noise and outliers in real applications. Experiments deliver superior performance over the previous state-of-the-arts on five real-world benchmarks, especially under the more difficult outlier scenarios, demon- strating the effectiveness of our method.`

## Supplementary File 
***
Supplementary materials related to our paper are available in [Supplementary_material.pdf](https://).

## Installation
***
Install the dependencies:
```
$ apt-get install ninja-build
$ conda create -n iassgm python=3.8
$ conda activate iassgm
$ conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
$ conda install pyg -c pyg
$ conda install -c dglteam/label/cu113 dgl
$ conda install scikit-learn
$ pip install numpy tqdm networkx PyGCL tensorboardX scipy easydict pyyaml
```
We recommand you to use docker.
```
$ docker pull 
```
## Available Datasets
### PascalVOC-Keypoint

1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like `data/PascalVOC/VOC2011` 
1. Download keypoint annotation for VOC2011 from [Berkeley server](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) or [google drive](https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR) and make sure it looks like `data/PascalVOC/annotations`
1. The train/test split is available in `data/PascalVOC/voc2011_pairs.npz`



Please cite the following papers if you use PascalVOC-Keypoint dataset:
```
@article{EveringhamIJCV10,
  title={The pascal visual object classes (voc) challenge},
  author={Everingham, Mark and Van Gool, Luc and Williams, Christopher KI and Winn, John and Zisserman, Andrew},
  journal={International Journal of Computer Vision},
  volume={88},
  pages={303–338},
  year={2010}
}

@inproceedings{BourdevICCV09,
  title={Poselets: Body part detectors trained using 3d human pose annotations},
  author={Bourdev, L. and Malik, J.},
  booktitle={International Conference on Computer Vision},
  pages={1365--1372},
  year={2009},
  organization={IEEE}
}
```

### Willow dataset

1. This dataset is available in [Willow-ObjectClass dataset](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip).
2. Unzip the dataset and make sure it looks like ``data/WillowObject/WILLOW-ObjectClass``

Please cite the following paper if you use Cars and Motorbikes dataset:
```
@inproceedings{ChoICCV13,
    author={Cho, Minsu and Alahari, Karteek and Ponce, Jean},
    title = {Learning Graphs to Match},
    booktitle = {International Conference on Computer Vision},
    pages={25--32},
    year={2013}
}
```

### CMU House/Hotel dataset

1. This dataset is available in `data/[Cmu-hotel-house](https://github.com/CMACH508/2021-IA-GM/tree/5d4b8ea5f8bba52622a0db7906d66ff830c652bc/data/Cmu-hotel-house)`.

Please cite the following paper if you use CMU House/Hotel dataset:
```
@article{caetano2009learning,
  title={Learning graph matching},
  author={Caetano, Tib{\'e}rio S and McAuley, Julian J and Cheng, Li and Le, Quoc V and Smola, Alex J},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={31},
  number={6},
  pages={1048--1058},
  year={2009},
  publisher={IEEE}
}
```

### CUB2011 dataset

1. Download [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz).
       
2.  Unzip the dataset and make sure it looks like ``data/CUB_200_2011/CUB_200_2011``

Please cite the following report if you use CUB2011 dataset:
```
@techreport{CUB2011,
Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
Year = {2011},
Institution = {California Institute of Technology},
Number = {CNS-TR-2011-001}
}
```

### IMC-PT-SparseGM dataset
1. Download the IMC-PT-SparseGM dataset from [google drive](https://drive.google.com/file/d/1Po9pRMWXTqKK2ABPpVmkcsOq-6K_2v-B/view?usp=sharing) or [baidu drive (code: 0576)](https://pan.baidu.com/s/1hlJdIFp4rkiz1Y-gztyHIw).
2. Unzip the dataset and make sure it looks like ``data/IMC_PT_SparseGM/annotations``
 Please cite the following papers if you use IMC-PT-SparseGM dataset:
```
@article{JinIJCV21,
    title={Image Matching across Wide Baselines: From Paper to Practice},
    author={Jin, Yuhe and Mishkin, Dmytro and Mishchuk, Anastasiia and Matas, Jiri and Fua, Pascal and Yi, Kwang Moo and Trulls, Eduard},
    journal={International Journal of Computer Vision},
    pages={517--547},
    year={2021}
}

@unpublished{WangPAMIsub21,
    title={Robust Self-supervised Learning of Deep Graph Matching with Mixture of Modes},
    author={Wang, Runzhong and Jiang, Shaofei and Yan, Junchi and Yang, Xiaokang},
    note={submitted to IEEE Transactions of Pattern Analysis and Machine Intelligence},
    year={2021}
}
```

## Data Preprocessing
***
```
$ cd data
$ python Pascal_voc_ssl.py
$ python willow_obj_ssl.py
$ python cmu_ssl.py
$ python cub2011_ssl.py
$ python imc_pt_sparsegm_ssl.py
```

We also provide the preprocessed dataset on the [google drive](https://).

## Run the Experiment
***
Run training and evaluation.
```bash
$ cd ..
$ python train_joint.py --cfg path/to/your/yaml
```

and replace ``path/to/your/yaml`` by path to your configuration file, e.g.
```bash
$ python train_joint.py --cfg experiments/gssl_joint_voc.yaml
```

Also, you can pretrain the GNN encoder.
```bash
$ python train_pretain.py --cfg path/to/your/yaml
```
Then finetune the model through
```bash
$ python train_finetune.py --cfg path/to/your/yaml
```
to try the two-step method mentioned in our paper.

Default configuration files are stored in``experiments/`` and you are welcomed to try your own configurations. If you find a better yaml configuration, please let us know by raising an issue or a PR and we will update the benchmark!
