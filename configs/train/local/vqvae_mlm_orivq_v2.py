exp_name = 'vqvae_mlm_v2'

pretrained_vq='/home/lr/models/vqvae/vqvae_youtube_d4_n2048_c512_embc512.pth'
# model settings
model = dict(
    type='Vqvae_Tracker_v2',
    backbone=dict(type='Vq_Swin',
                  res_blocks=dict(depth=10, inplanes=128),
                  vqvae=dict(type='VQVAE', downsample=4, n_embed=2048, channel=512, n_res_channel=128, embed_dim=512),
                  pretrained_vq=pretrained_vq
    ),
    vqvae=dict(type='VQVAE', downsample=4, n_embed=2048, channel=512, n_res_channel=128, embed_dim=512),
    ce_loss=dict(type='Ce_Loss',reduction='none'),
    l2_loss = None,
    patch_size=-1,
    fc=True,
    temperature=0.1,
    pretrained_vq=pretrained_vq,
    pretrained=None
)

# model training and testing settings
train_cfg = dict(syncbn=False)
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
train_dataset_type = 'VOS_youtube_dataset_mlm_motion'

val_dataset_type = None
test_dataset_type = 'VOS_davis_dataset_test'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

# img_norm_cfg = dict(
#     mean=[0, 0, 0], std=[255, 255, 255], to_bgr=False)


train_pipeline = [
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
        p=0.8,
        same_across_clip=False,
        same_on_clip=False,
        output_keys='jitter_imgs'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Normalize', **img_norm_cfg, keys='jitter_imgs'),
    dict(type='FormatShape', input_format='NPTCHW'),
    dict(type='FormatShape', input_format='NPTCHW', keys='jitter_imgs'),
    dict(type='Collect', keys=['imgs', 'jitter_imgs', 'mask_query_idx'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'jitter_imgs', 'mask_query_idx'])
]

val_pipeline = [
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
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
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=
            dict(
            type=train_dataset_type,
            size=256,
            p=0.7,
            root='/home/lr/dataset/YouTube-VOS',
            list_path='/home/lr/dataset/YouTube-VOS/2018/train',
            data_prefix='2018',
            mask_ratio=0.15,
            clip_length=2,
            vq_size=32,
            pipeline=train_pipeline,
            test_mode=False),

    # test =  dict(
    #         type=train_dataset_type,
    #         root='/home/lr/dataset/YouTube-VOS',
    #         list_path='/home/lr/dataset/YouTube-VOS/2018/train',
    #         data_prefix='2018',
    #         mask_ratio=0.15,
    #         clip_length=2,
    #         vq_size=32,
    #         pipeline=train_pipeline,
    #         test_mode=True)

    test =  dict(
            type=test_dataset_type,
            root='/home/lr/dataset/DAVIS',
            list_path='/home/lr/dataset/DAVIS/ImageSets',
            data_prefix='2017',
            pipeline=val_pipeline,
            test_mode=True
             )
)

# optimizer
optimizers = dict(
    backbone=dict(type='Adam', lr=0.001, betas=(0.9, 0.999)),
    predictor=dict(type='Adam', lr=0.001, betas=(0.9, 0.999)),
    )

# learning policy
# total_iters = 200000
runner_type='epoch'
max_epoch=200
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.01,
    by_epoch=False,
    # warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.1,
    warmup_by_epoch=True
    )

checkpoint_config = dict(interval=1, save_optimizer=True, by_epoch=True)


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
work_dir = f'./output/{exp_name}'

load_from = None
resume_from = None
workflow = [('train', 1)]