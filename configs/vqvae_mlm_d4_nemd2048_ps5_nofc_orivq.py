exp_name = 'vqvae_mlm_d4_nemd2048_ps5_fc_orivq'

# model settings
model = dict(
    type='Vqvae_Tracker',
    backbone=dict(type='ResNet',depth=18, strides=(1, 2, 1, 1), out_indices=(3, )),
    vqvae=dict(type='VQVAE',downsample=4, n_embed=2048),
    ce_loss=dict(type='Ce_Loss',reduction='none'),
    patch_size=5,
    fc=False,
    pretrained='/gdata/lirui/models/vqvae/vqvae_d4_n2048.pth'
)

# model training and testing settings
train_cfg = None
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
train_dataset_type = 'VOS_youtube_dataset_mlm'

val_dataset_type = None
test_dataset_type = 'VOS_davis_dataset_test'


# train_pipeline = None
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='RandomResizedCrop', area_range=(0.6,1.0)),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NPTCHW'),
    dict(type='Collect', keys=['imgs', 'mask_query_idx'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'mask_query_idx'])
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
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=16, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=
            dict(
            type=train_dataset_type,
            root='/gdata/lirui/dataset/YouTube-VOS',
            list_path='/gdata/lirui/dataset/YouTube-VOS/2018/train',
            data_prefix='2018',
            mask_ratio=0.15,
            clip_length=3,
            vq_size=32,
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
)

# optimizer
optimizers = dict(
        type='Adam',
        lr=0.001,
        betas=(0.9, 0.999),
        )

# learning policy
# total_iters = 200000
runner_type='epoch'
max_epoch=200
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.001,
    by_epoch=False
    )

checkpoint_config = dict(interval=50, save_optimizer=True, by_epoch=True)
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
work_dir = f'/gdata/lirui/expdir/VCL/group_vqvae_tracker/{exp_name}'

load_from = None
resume_from = None
workflow = [('train', 1)]
