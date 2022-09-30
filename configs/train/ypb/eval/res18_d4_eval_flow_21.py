import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
from vcl.utils import *

exp_name = 'res18_d4_eval_flow'
docker_name = 'bit:5000/lirui_torch1.8_cuda11.1_corres'

# model settings
model = dict(
    type='Memory_Tracker_Flow',
            num_levels=4,
            cxt_channels=128,
            h_channels=128,
            flow_clamp=2,
            corr_op_cfg=dict(type='CorrLookup', align_corners=True, radius=2),
            corr_op_cfg_infer=dict(type='CorrLookup_Infer', align_corners=True, radius=6),
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
                # flow_loss=dict(type='SequenceLoss'),
                act_cfg=dict(type='ReLU')),
            loss=dict(type='SequenceLoss'),
            loss_weight=dict(flow_rec_loss=1),
            drop_ch=False,
            freeze_bn=False
)

model_test = None

# model training and testing settings
train_cfg = dict(syncbn=True)

test_cfg = dict(
    zero_flow=False,
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
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='RGB2LAB', output_keys='images_lab'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Normalize', **img_norm_cfg_lab, keys='images_lab'),
    # dict(type='ColorDropout', keys='jitter_imgs', drop_rate=0.8),
    dict(type='FormatShape', input_format='NPTCHW'),
    dict(type='FormatShape', input_format='NPTCHW', keys='images_lab'),
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
    train_dataloader=dict(samples_per_gpu=8, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=
            dict(
            type=train_dataset_type,
            root='/gdata/lirui/dataset/YouTube-VOS',
            list_path='/gdata/lirui/dataset/YouTube-VOS/2018/train',
            data_prefix=dict(RGB='train/JPEGImages_s256', FLOW='train_all_frames/Flows_s256', ANNO='train/Annotations'),
            clip_length=2,
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
optimizers = dict(
    backbone=dict(type='Adam', lr=0.001, betas=(0.9, 0.999))
    )
# learning policy
# total_iters = 200000
runner_type='epoch'
max_epoch=1600
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.001,
    by_epoch=False,
    warmup_iters=10,
    warmup_ratio=0.1,
    warmup_by_epoch=True
    )

checkpoint_config = dict(interval=1600, save_optimizer=True, by_epoch=True)
# remove gpu_collect=True in non distributed training
# evaluation = dict(interval=1000, save_image=False, gpu_collect=False)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False, interval=10),
        # dict(type='WandbLoggerHook', init_kwargs=dict(project='video_correspondence', name=f'{exp_name}'))
    ])

visual_config = None


# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'/gdata/lirui/expdir/VCL/group_stsl/{exp_name}'


evaluation = dict(output_dir=f'{work_dir}/eval_output_val', interval=800, by_epoch=True
                  )

eval_config= dict(
                  output_dir=f'{work_dir}/eval_output',
                  checkpoint_path='/gdata/lirui/expdir/VCL/group_fm_flow/spa_temp_d4_r2_raft_test/epoch_160.pth',
                  torchvision_pretrained=None
                )


load_from = None
resume_from = None
# ddp_shuffle = True
workflow = [('train', 1)]
find_unused_parameters = True
test_mode = True



if __name__ == '__main__':

    make_local_config(exp_name, file='eval')