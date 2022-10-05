import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
from vcl.utils import *

exp_name = 'spa_temp_r2_d4_raft_flowsup'
docker_name = 'bit:5000/lirui_torch1.8_cuda11.1_corres'

# model settings
model = dict(
            type='Memory_Tracker_Flow',
            num_levels=4,
            cxt_channels=128,
            h_channels=128,
            # flow_clamp=2,
            corr_op_cfg=dict(type='CorrLookup', align_corners=True, radius=2),
            corr_op_cfg_infer=dict(type='CorrLookup_Infer', align_corners=True, radius=2),
            backbone=dict(type='ResNet',depth=18, strides=(1, 2, 2, 1), out_indices=(2, ), pool_type='none'),
            cxt_backbone=dict(
                type='RAFTEncoder',
                in_channels=3,
                out_channels=256,
                net_type='Basic',
                # norm_cfg=dict(type='SyncBN'),
                init_cfg=[
                    dict(
                        type='Kaiming',
                        layer=['Conv2d'],
                        mode='fan_out',
                        nonlinearity='relu'),
                    dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
                ]),
            decoder=dict(
                type='RAFTDecoder',
                net_type='Basic',
                num_levels=4,
                radius=4,
                iters=6,
                mask_pred=False,
                corr_op_cfg=dict(type='CorrLookup', align_corners=True),
                gru_type='SeqConv',
                act_cfg=dict(type='ReLU')),
            target_model=dict(
                            type='RAFT',
                            num_levels=4,
                            radius=4,
                            cxt_channels=128,
                            h_channels=128,
                            backbone=dict(
                                type='RAFTEncoder',
                                in_channels=3,
                                out_channels=256,
                                net_type='Basic',
                                norm_cfg=dict(type='IN'),
                                init_cfg=[
                                    dict(
                                        type='Kaiming',
                                        layer=['Conv2d'],
                                        mode='fan_out',
                                        nonlinearity='relu'),
                                    dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)
                                ]),
                            cxt_backbone=dict(
                                type='RAFTEncoder',
                                in_channels=3,
                                out_channels=256,
                                net_type='Basic',
                                # norm_cfg=dict(type='SyncBN'),
                                init_cfg=[
                                    dict(
                                        type='Kaiming',
                                        layer=['Conv2d'],
                                        mode='fan_out',
                                        nonlinearity='relu'),
                                    dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
                                ]),
                            decoder=dict(
                                type='RAFTDecoder',
                                net_type='Basic',
                                num_levels=4,
                                radius=4,
                                iters=30,
                                corr_op_cfg=dict(type='CorrLookup', align_corners=True),
                                gru_type='SeqConv',
                                act_cfg=dict(type='ReLU')),
                            freeze_bn=True,
                            loss=dict(type='SequenceLoss'),
                            init_cfg=dict(type='Pretrained', checkpoint='/home/lr/models/optical_flow/raft_8x2_100k_mixed_368x768.pth')
                                ),
            loss=dict(type='SequenceLoss'),
            loss_weight=dict(flow_rec_loss=0, raft_gt_loss=0.01),
            drop_ch=False,
            freeze_bn=False
)


model_test = None

# model training and testing settings
train_cfg = dict(syncbn=True)

test_cfg = dict(
    precede_frames=20,
    topk=10,
    temperature=0.07,
    strides=(1, 2, 2, 1),
    out_indices=(3, ),
    neighbor_range=24,
    with_first=True,
    with_first_neighbor=True,
    output_dir='eval_results')

# dataset settings
train_dataset_type = 'VOS_youtube_dataset_rgb'

val_dataset_type = 'VOS_davis_dataset_test'
test_dataset_type = 'VOS_davis_dataset_test'


# train_pipeline = None
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
img_norm_cfg_lab = dict(mean=[50, 0, 0], std=[50, 127, 127], to_bgr=False)

train_pipeline = [
    dict(type='RandomResizedCrop', area_range=(0.6,1.0), aspect_ratio_range=(1.5, 2.0),),
    dict(type='Resize', scale=(384, 384), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='RGB2LAB', output_keys='images_lab'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Normalize', **img_norm_cfg_lab, keys='images_lab'),
    # dict(type='ColorDropout', keys='jitter_imgs', drop_rate=0.8),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='FormatShape', input_format='NCTHW', keys='images_lab'),
    dict(type='Collect', keys=['imgs', 'images_lab'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'images_lab'])
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

# demo_pipeline = None
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=
            dict(
            type='RepeatDataset',
            dataset=  dict(
                    type=train_dataset_type,
                    root='/home/lr/dataset/YouTube-VOS',
                    list_path='/home/lr/dataset/YouTube-VOS/2018/train',
                    data_prefix=dict(RGB='train/JPEGImages_s256', FLOW=None, ANNO=None),
                    clip_length=2,
                    # steps=dict(v=[1,3,5], p=[0.5,0.3,0.2]),
                    pipeline=train_pipeline,
            test_mode=False),
            times=10,
            ),
          

    test =  dict(
            type=test_dataset_type,
            root='/home/lr/dataset/DAVIS',
            list_path='/home/lr/dataset/DAVIS/ImageSets',
            data_prefix='2017',
            pipeline=val_pipeline,
            test_mode=True
            ),
    
    val =  dict(
            type=val_dataset_type,
            root='/home/lr/dataset/DAVIS',
            list_path='/home/lr/dataset/DAVIS/ImageSets',
            data_prefix='2017',
            pipeline=val_pipeline,
            test_mode=True
            ),
)
# optimizer
optimizers = dict(
    type='AdamW',
    lr=0.0005,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0001,
    amsgrad=False)
# learning policy
# total_iters = 200000
runner_type='epoch'
max_epoch=160
lr_config = dict(
    policy='OneCycle',
    max_lr=0.0005,
    pct_start=0.05,
    anneal_strategy='linear')


checkpoint_config = dict(interval=max_epoch//2, save_optimizer=True, by_epoch=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False, interval=10),
    ])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'/home/lr/expdir/VCL/group_fm_flow/{exp_name}'

visual_config = dict(type='VisualizationHook', interval=50, res_name_list=['flow', 'imgs'], output_dir=work_dir+'/vis')


# evaluation = dict(output_dir=f'{work_dir}/eval_output_val', interval=max_epoch//2, by_epoch=True
#                   )

eval_config= dict(
                  output_dir=f'{work_dir}/eval_output',
                  checkpoint_path=f'/home/lr/expdir/VCL/group_fm_flow/{exp_name}/epoch_{max_epoch}.pth'
                )


load_from = None
resume_from = None
ddp_shuffle = True
workflow = [('train', 1)]
find_unused_parameters = True



if __name__ == '__main__':

    make_local_config(exp_name, file='fm_flow')