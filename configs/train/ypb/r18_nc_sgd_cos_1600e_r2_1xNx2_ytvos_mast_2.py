import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from vcl.utils import *

exp_name = 'r18_nc_sgd_cos_1600e_r2_1xNx2_ytvos_mast_2'
docker_name = 'bit:5000/lirui_torch1.8_cuda11.1_corres'

model = dict(
    type='SimaSiam_Rec',
    backbone=dict(
        type='ResNet',
        pretrained=None,
        depth=18,
        out_indices=(2, 3, ),
        strides=(1,2,1,2),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        zero_init_residual=True,
        pool_type='mean'),
    img_head=dict(
        type='SimSiamHead',
        in_channels=512,
        norm_cfg=dict(type='SyncBN'),
        num_projection_fcs=3,
        projection_mid_channels=128,
        projection_out_channels=128,
        num_predictor_fcs=2,
        predictor_mid_channels=128,
        predictor_out_channels=128,
        with_norm=True,
        loss_feat=dict(type='CosineSimLoss', negative=False),
        spatial_type='avg'),
    mask_radius=6,
    scaling_att=True
    )

model_test = dict(
    type='VanillaTracker',
    backbone=dict(type='ResNet',depth=18, strides=(1, 2, 1, 1), out_indices=(2, ), pool_type='mean'),
)


# model training and testing settings
train_cfg = dict(intra_video=True)
test_cfg = dict(
    precede_frames=20,
    topk=10,
    temperature=0.07,
    strides=(1, 2, 1, 1),
    out_indices=(2, ),
    neighbor_range=24,
    with_first=True,
    with_first_neighbor=True,
    output_dir='eval_results')

# dataset settings
train_dataset_type = 'VOS_youtube_dataset_rgb_withbbox_V2'

val_dataset_type = 'VOS_davis_dataset_test'
test_dataset_type = 'VOS_davis_dataset_test'


# train_pipeline = None
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
img_norm_cfg_lab = dict(mean=[50, 0, 0], std=[50, 127, 127], to_bgr=False)


train_pipeline = [
    dict(type='RandomResizedCrop', area_range=(0.2,1.0), aspect_ratio_range=(1.5, 2.0),same_across_clip=False,same_on_clip=False, keys='imgs_spa_aug', with_bbox=True, crop_ratio=0.7),
    dict(type='RandomResizedCrop', area_range=(0.6,1.0), aspect_ratio_range=(1.5, 2.0), same_across_clip=True,same_on_clip=True,),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Resize', scale=(256, 256), keep_ratio=False, keys='imgs_spa_aug'),
    dict(type='Flip', flip_ratio=0.5,same_across_clip=False,same_on_clip=False,  keys='imgs_spa_aug'),
    dict(type='Flip', flip_ratio=0.5, same_across_clip=True,same_on_clip=True,),
    dict(type='RGB2LAB'),
    dict(type='RGB2LAB',keys='imgs_spa_aug', output_keys='imgs_spa_aug'), 
    dict(type='Normalize', **img_norm_cfg_lab),
    dict(type='Normalize', **img_norm_cfg_lab, keys='imgs_spa_aug'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='FormatShape', input_format='NCTHW', keys='imgs_spa_aug'),
    dict(type='Collect', keys=['imgs','imgs_spa_aug'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'imgs_spa_aug'])
]


val_pipeline = [
    dict(type='Resize', scale=(-1, 480), keep_ratio=True),
    dict(type='Flip', flip_ratio=0),
    dict(type='RGB2LAB'),
    dict(type='Normalize', **img_norm_cfg_lab),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'ref_seg_map'],
        meta_keys=('video_path', 'original_shape')),
    dict(type='ToTensor', keys=['imgs', 'ref_seg_map'])
]

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
            data_prefix=dict(RGB='train/JPEGImages_s256', FLOW='train/Flows_s256', ANNO='train/Annotations'),
            num_clips=2,
            clip_length=1,
            pipeline=train_pipeline,
            test_mode=False),

    test =  dict(
            type=test_dataset_type,
            root='/gdata/lirui/dataset/DAVIS',
            list_path='/gdata/lirui/dataset/DAVIS/ImageSets',
            data_prefix='2017',
            pipeline=val_pipeline,
            test_mode=True
            ),
    
    val =  dict(
            type=val_dataset_type,
            root='/gdata/lirui/dataset/DAVIS',
            list_path='/gdata/lirui/dataset/DAVIS/ImageSets',
            data_prefix='2017',
            pipeline=val_pipeline,
            test_mode=True
            ),
)



# optimizer
optimizers = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
# optimizers = dict(
#     backbone=dict(type='Adam', lr=0.001, betas=(0.9, 0.999))
#     )

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
runner_type='epoch'
max_epoch = 1600
checkpoint_config = dict(interval=800, save_optimizer=True, by_epoch=True)
# evaluation = dict(
#     interval=1,
#     metrics='davis',
#     key_indicator='feat_1.J&F-Mean',
#     rule='greater')
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False, interval=10),
    ])


# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'/gdata/lirui/expdir/VCL/group_vqvae_tracker/{exp_name}'


evaluation = dict(output_dir=f'{work_dir}/eval_output_val', interval=800, by_epoch=True
                  )
eval_config= dict(
                  output_dir=f'{work_dir}/eval_output',
                  checkpoint_path=f'/gdata/lirui/expdir/VCL/group_vqvae_tracker/{exp_name}/epoch_{max_epoch}.pth'
                )


load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = False


if __name__ == '__main__':
    make_pbs(exp_name, docker_name)
    make_local_config(exp_name)