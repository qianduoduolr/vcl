import _init_paths
import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist

from vcl.apis import set_random_seed, train_model
from vcl.datasets import build_dataset
from vcl.models import build_model
from vcl.utils import collect_env, get_root_logger



def main():
    
    # edit whatever you want test
    
    vqvae = build_model(dict(type='VQVAE_3D'))
    data = torch.ones((1, 3, 4, 256, 256))
    out = vqvae(data)
    
    
    




if __name__ == '__main__':
    main()