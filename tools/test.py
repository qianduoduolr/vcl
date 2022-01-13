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
from vcl.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train an editor')
    parser.add_argument('--config', help='train config file path', default='/home/lr/project/vcl/configs/test/vqvae_mlm_orivq_viz.py')
    # parser.add_argument('--checkpoint', type=str, help='checkpoint file', default='')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file', default='/home/lr/expdir/VCL/group_vqvae_tracker/vqvae_mlm_d4_nemd2048_byol_dyt_nl_l5_fc_orivq_withbbox_random_v2_longterm/epoch_3200.pth')
    parser.add_argument('--seed', type=int, default=4, help='random seed')
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
        default='output/youtube/vis_vq_vqvae_mlm_d4_nemd2048_byol_dyt_nl_l5_fc_orivq_withbbox_random_v2_longterm',
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
    rank, _ = get_dist_info()

    logger = get_root_logger()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    # Overwrite eval_config from args.eval
    eval_config = merge_configs(eval_config, dict(metrics=args.eval))

    if args.out:
        eval_config['output_dir'] = os.path.join(args.out, 'eval_output')
    if 'output_dir' in eval_config:
        args.tmpdir = eval_config['output_dir']
    if 'checkpoint_path' in eval_config:
        args.checkpoint = eval_config['checkpoint_path']
        eval_config.pop('checkpoint_path')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # model.backbone.pretrained = '/home/lr/models/ssl/vcl/vfs_pretrain/r18_nc_sgd_cos_100e_r2_1xNx8_k400-db1a4c0d.pth'
    # model.backbone.init_weights()

    args.save_image = args.save_path is not None
    empty_cache = cfg.get('empty_cache', False)

    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
    }

    data_loader = build_dataloader(dataset, **loader_cfg)

    if args.checkpoint:
        _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
        logger.info('load pretrained model successfully')

    model = MMDataParallel(model, device_ids=[0])

    args.save_path = os.path.join(args.save_path, f'out{model.module.backbone.out_indices[0]}')
    outputs = single_gpu_test(
            model,
            data_loader,
            save_path=args.save_path,
            save_image=args.save_image)
    
    if outputs[0] != None:
        import numpy as np

        predict = []
        target = []
        for out in outputs:
            target.append(out[0])
            predict.append(out[1])

        predict = np.concatenate(predict,0).reshape(-1)
        target = np.concatenate(target,0).reshape(-1)

        #######################
        import matplotlib.pyplot as plt

        def vis_bar(inp, name):
            plt.figure()

            with open(os.path.join(args.save_path, 'result.txt'), 'a') as f:    
                print('There are {} keys in {}'.format(len(np.unique(inp)), name))
                f.write('There are {} keys in {}'.format(len(np.unique(inp)), name) + '\n')

            x = np.array(list([ (i==inp).astype(np.uint8).sum() for i in range(2048) ]))

            # the histogram of the data
            plt.bar(range(2048), x)

            plt.xlabel('keys')
            plt.ylabel('times')
            plt.title('Histogram of times of each key ')

            # Tweak spacing to prevent clipping of ylabel
            # plt.savefig(f'./test_{name}.jpg')
            # plt.show()

        vis_bar(predict, 'predict')
        vis_bar(target, 'target')


        ##########################
        with open(os.path.join(args.save_path, 'result.txt'), 'a') as f:    
            print('predict acc is {:4f}'.format((predict==target).astype(np.uint8).sum() / len(predict)))
            f.write('predict acc is {:4f}'.format((predict==target).astype(np.uint8).sum() / len(predict)))

if __name__ == '__main__':
    main()
