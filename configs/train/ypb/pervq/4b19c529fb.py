import os
from random import sample
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from vcl.utils import *

exp_name = '4b19c529fb'
docker_name = 'bit:5000/lirui_torch1.8_cuda11.1_corres'

# model settings
model = dict(
    type='VQCL_v2',
    backbone=dict(type='ResNet', depth=18, strides=(1, 2, 1, 1), out_indices=(3, )), #pretrained='/gdata/lirui/models/ssl/vcl/vfs_pretrain/r18_nc_sgd_cos_100e_r2_1xNx8_k400-db1a4c0d.pth'),
    sim_siam_head=dict(
        type='SimSiamHead',
        in_channels=512,
        # norm_cfg=dict(type='SyncBN'),
        num_projection_fcs=3,
        projection_mid_channels=512,
        projection_out_channels=512,
        num_predictor_fcs=2,
        predictor_mid_channels=128,
        predictor_out_channels=512,
        with_norm=True,
        spatial_type='avg'),
    loss=dict(type='CosineSimLoss', negative=False),
    embed_dim=128,
    n_embed=32,
    commitment_cost=0.25,
    decay=0.5,
    pretrained='/gdata/lirui/models/vqvae/vqvae_youtube_d4_n32_c256_embc128_byol_commit1.0.pth'
)
# model = dict(
#     type='VQVAE',
#     downsample=4, 
#     n_embed=32, 
#     channel=256, 
#     n_res_channel=128, 
#     embed_dim=128,
#     commitment_cost=1,
#     loss=dict(type='MSELoss',reduction='mean'),
#     pretrained='/gdata/lirui/models/vqvae/vqvae_youtube_d4_n32_c256_embc128.pth'
# )



# model training and testing settings
train_cfg = dict(syncbn=True)

test_cfg = dict(
    precede_frames=20,
    topk=10,
    temperature=0.07,
    strides=(1, 2, 1, 1),
    out_indices=(3, ),
    neighbor_range=24,
    with_first=True,
    with_first_neighbor=True,
    output_dir='eval_results')

# dataset settings
train_dataset_type = 'VOS_youtube_dataset_rgb'

val_dataset_type = None
test_dataset_type = 'VOS_davis_dataset_test'


# train_pipeline = None
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='RandomResizedCrop', area_range=(0.6,1.0), aspect_ratio_range=(1.5, 2.0),),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NPTCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

val_pipeline = [
    dict(type='Resize', scale=(-1, 480), keep_ratio=True),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'ref_seg_map'],
        meta_keys=('video_path', 'original_shape')),
    dict(type='ToTensor', keys=['imgs', 'ref_seg_map'])
]

# demo_pipeline = None
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=32, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=
            dict(
            type=train_dataset_type,
            root='/dev/shm',
            list_path='/dev/shm/2018/train',
            data_prefix=dict(RGB='train/JPEGImages_s256', ANNO='train/Annotations'),
            clip_length=1,
            num_clips=2,
            pipeline=train_pipeline,
            per_video='4b19c529fb'
            ),

    test =  dict(
            type=test_dataset_type,
            root='/gdata/lirui/dataset/DAVIS',
            list_path='/gdata/lirui/dataset/DAVIS/ImageSets',
            data_prefix='2017',
            pipeline=val_pipeline,
            test_mode=True
            ),
)

# optimizer
# optimizers = dict(type='Adam', lr=3e-4, betas=(0.9, 0.999))
optimizers = dict(type='SGD', lr=0, momentum=0.9, weight_decay=0.0001)


# learning policy
# total_iters = 200000
runner_type='epoch'
max_epoch=1
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.001,
    by_epoch=False,
    warmup_iters=10,
    warmup_ratio=0.1,
    warmup_by_epoch=True
    )

checkpoint_config = dict(interval=1, save_optimizer=True, by_epoch=True)
# remove gpu_collect=True in non distributed training
# evaluation = dict(interval=1000, save_image=False, gpu_collect=False)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False, interval=10),
    ])

visual_config = None


# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'/gdata/lirui/expdir/VCL/group_vqvae_tracker/per_video_vq/{exp_name}'

eval_config= dict(
                  output_dir=f'{work_dir}/eval_output',
                  checkpoint_path=f'/gdata/lirui/expdir/VCL/group_vqvae_tracker/per_video_vq/{exp_name}/epoch_{max_epoch}.pth',
                  dry_run=True
                )


load_from = None
resume_from = None
workflow = [('train', 1)]



if __name__ == '__main__':
    make_pbs(exp_name, docker_name)
    make_local_config(exp_name)