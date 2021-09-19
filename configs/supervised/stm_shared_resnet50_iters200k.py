exp_name = 'stm_shared_resnet50_iters200k'

# model settings
model = dict(
    type='STM',
    depth=18,
    pixel_loss=dict(type='Pixel_Ce_Loss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
train_cfg = None
test_cfg = dict(test_memory_every_frame=5, memory_num=None, metrics=['JFM'])

# dataset settings
train_dataset_type1 = 'VOS_youtube_dataset'
train_dataset_type2 = 'VOS_davis_dataset'

val_dataset_type = 'VOS_davis_dataset'
test_dataset_type = 'VOS_davis_dataset'


# train_pipeline = None

# val_pipeline = None

# test_pipeline = None

# demo_pipeline = None

data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=
        [
            dict(
            type=train_dataset_type1,
            root='/home/lr/dataset/YouTube-VOS/train',
            pipeline=None,
            test_mode=False),

            dict(
            type=train_dataset_type2,
            root='/home/lr/dataset/DAVIS',
            imset='2017/train.txt',
            resolution='480p',
            single_object=False,
            pipeline=None,
            test_mode=False),
        ],
    
    # val
    val =  dict(
            type=val_dataset_type,
            root='/home/lr/dataset/DAVIS',
            imset='2017/val.txt',
            resolution='480p',
            single_object=False,
            pipeline=None,
            test_mode=True
            ),

    # test
    test =  dict(
            type=test_dataset_type,
            root='/home/lr/dataset/DAVIS',
            imset='2017/val.txt',
            resolution='480p',
            single_object=False,
            pipeline=None,
            test_mode=True
            ),
)

# optimizer
optimizers = dict(
        type='Adam',
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8
        )

# learning policy
total_iters = 200000
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.01,
    by_epoch=False
    )

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=1000, save_image=False, gpu_collect=False)
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
work_dir = f'./work_dirs/{exp_name}'
# load_from = '/home/lr/models/segmentation/coco_pretrained_resnet50_679999_169.pth'
load_from = None
resume_from = None
workflow = [('train', 1)]
