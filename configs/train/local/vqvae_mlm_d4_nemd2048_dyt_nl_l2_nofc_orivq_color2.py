import os
exp_name = 'vqvae_mlm_d4_nemd2048_dyt_nl_l2_nofc_orivq_color2'
docker_name = 'bit:5000/lirui_torch1.5_cuda10.1_corr'

# model settings
model = dict(
    type='Vqvae_Tracker',
    backbone=dict(type='ResNet',depth=18, strides=(1, 2, 1, 1), out_indices=(3, )),
    vqvae=dict(type='VQVAE', downsample=4, n_embed=2048, channel=256, n_res_channel=128, embed_dim=128),
    ce_loss=dict(type='Ce_Loss',reduction='none'),
    patch_size=-1,
    fc=False,
    temperature=0.1,
    pretrained_vq='/home/lr/models/vqvae/vqvae_youtube_d4_n2048_c256_embc128',
    pretrained=None
)

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
train_dataset_type = 'VOS_youtube_dataset_mlm'

val_dataset_type = None
test_dataset_type = 'VOS_davis_dataset_test'


# train_pipeline = None
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
# img_norm_cfg = dict(
#     mean=[0, 0, 0], std=[255, 255, 255], to_bgr=False)

train_pipeline = [
    dict(type='RandomResizedCrop', area_range=(0.2,1.0)),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
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
    train_dataloader=dict(samples_per_gpu=5, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=
            dict(
            type=train_dataset_type,
            root='/home/lr/dataset/YouTube-VOS',
            list_path='/home/lr/dataset/YouTube-VOS/2018/train',
            data_prefix='2018',
            mask_ratio=0.15,
            clip_length=2,
            vq_size=32,
            pipeline=train_pipeline,
            test_mode=False),

    test =  dict(
            type=test_dataset_type,
            root='/home/lr/dataset/DAVIS',
            list_path='/home/lr/dataset/DAVIS/ImageSets',
            data_prefix='2017',
            pipeline=val_pipeline,
            test_mode=True
            ),
)

# optimizer
optimizers = dict(
    backbone=dict(type='Adam', lr=0.001, betas=(0.9, 0.999)),
    embedding_layer=dict(type='Adam', lr=0.001, betas=(0.9, 0.999))
    )
# learning policy
# total_iters = 200000
runner_type='epoch'
max_epoch=800
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.001,
    by_epoch=False,
    warmup_iters=10,
    warmup_ratio=0.1,
    warmup_by_epoch=True
    )

checkpoint_config = dict(interval=200, save_optimizer=True, by_epoch=True)
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
work_dir = f'/home/lr/expdir/VCL/group_vqvae_tracker/{exp_name}'

eval_config= dict(
                  output_dir=f'{work_dir}/eval_output',
                  checkpoint_path=f'/home/lr/expdir/VCL/group_vqvae_tracker/{exp_name}/epoch_{max_epoch}.pth'
                )


load_from = None
resume_from = None
workflow = [('train', 1)]


def make_pbs():
    pbs_data = ""
    with open('configs/pbs/template.pbs', 'r') as f:
        for line in f:
            line = line.replace('exp_name',f'{exp_name}')
            line = line.replace('docker_name', f'{docker_name}')
            pbs_data += line

    with open(f'configs/pbs/{exp_name}.pbs',"w") as f:
        f.write(pbs_data)

def make_local_config():
    config_data = ""
    with open(f'configs/train/local/{exp_name}.py', 'r') as f:
        for line in f:
            line = line.replace('/home/lr','/gdata/lirui')
            # line = line.replace('/home/lr/dataset','/home/lr/dataset')
            config_data += line

    with open(f'configs/train/ypb/{exp_name}.py',"w") as f:
        f.write(config_data)


if __name__ == '__main__':
    make_pbs()
    make_local_config()