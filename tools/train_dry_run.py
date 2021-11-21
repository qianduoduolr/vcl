# Copyright (c) OpenMMLab. All rights reserved.
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train an editor')
    parser.add_argument('--config', help='train config file path', default='/home/lr/project/vcl/configs/train/local/vqvae_mlm_d4_nemd8_dyt_nl_l2_fc_orivq_motion.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        default=False,
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()


if __name__ == '__main__':
    main()