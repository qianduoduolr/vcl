# Copyright (c) OpenMMLab. All rights reserved.
import _init_paths
import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from vcl.apis import multi_gpu_test, set_random_seed, single_gpu_test
from vcl.core.distributed_wrapper import DistributedDataParallelWrapper
from vcl.datasets import build_dataloader, build_dataset
from vcl.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('--config', help='test config file path', default='/home/lr/project/vcl/configs/train/local/vqvae_mlm_v2_d4_nemd2048_dyt_nl_fc_orivq_motion2.py')
    # parser.add_argument('--checkpoint', type=str, help='checkpoint file', default='/home/lr/expdir/VCL/group_vqvae_tracker/vqvae_mlm_d4_nemd2048_dyt_nl_l2_nofc_orivq/epoch_800.pth')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file', default='')
    parser.add_argument('--out-indices', nargs='+', type=int, default=[0])
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--out', type=str, help='output result file', default='')
    parser.add_argument(
        '--eval',
        type=str,
        default='davis',
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')

    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--save-path',
        default=None,
        type=str,
        help='path to store images and if not given, will not save image')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1

def main():
    args = parse_args()

if __name__ == '__main__':
    main()