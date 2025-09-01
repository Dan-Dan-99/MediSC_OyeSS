_base_ = ['../configs/rtmdet/rtmdet_m_8xb32-300e_coco.py']

pretrained = 'configs/swin_large_patch4_window12_384_22k.pth'

dataset_type = 'CocoDataset'
data_root = 'data/hand_tool_dataset'
backend_args = dict(_delete_=True, backend='local')

metainfo = dict(
    classes=('Left Hand', 'Right Hand'),
    palette=[(255, 0, 0), (128, 0, 128)]
)

model = dict(
    bbox_head=dict(
        num_classes=2,
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        )
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False
    ),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5
    ),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1536, 864), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# DataLoader 설정
train_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train_hands.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val_hands.json',
        data_prefix=dict(img='val/'),
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True
    )
)

test_dataloader = val_dataloader

# 최적화된 학습 설정
base_lr = 0.0001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True
    )
)

max_epochs = 100
num_last_epochs = 15
interval = 5

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49
    ),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - num_last_epochs,
        switch_pipeline=train_pipeline
    )
]

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/hand_tool_dataset/annotations/val_hands.json',
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='DetLocalVisualizer',
#     vis_backends=vis_backends,
#     name='visualizer'
# )
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='oss_task3',
                settings=dict(
                    allow_val_change=True  # 값 변경 허용
                )
            )
        ),
        dict(type='LocalVisBackend')
    ]
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)
val_cfg = dict(_delete_=True, type='ValLoop')
test_cfg = dict(_delete_=True, type='TestLoop')