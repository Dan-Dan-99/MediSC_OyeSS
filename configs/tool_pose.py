# 1. General Configuration
_base_ = ['_base_/default_runtime_pose.py']

default_scope = 'mmpose'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)

log_level = 'INFO'
load_from = None
resume = False

dataset_type = 'CocoDataset'
data_root = 'data/hand_tool_dataset/'

dataset_info = dict(
    dataset_name='surgical_tools',
    paper_info=dict(),
    keypoint_info={
        0: dict(name='Left', id=0, color=[255, 0, 0], type='', swap=''),
        1: dict(name='Right', id=1, color=[255, 128, 0], type='', swap=''),
        2: dict(name='Joint', id=2, color=[255, 255, 0], type='', swap=''),
    },
    skeleton_info={},
    num_keypoints=3,
    joint_weights=[1.0] * 3,
    sigmas=[0.025] * 3
)

# Data pipeline
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='RandomHalfBody', prob=0.3),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.6, 1.4],
        rotate_factor=80),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='GenerateTarget', encoder=dict(
        type='MSRAHeatmap',
        input_size=(256, 256),
        heatmap_size=(64, 64),
        sigma=2)),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs')
]

# Dataloader 설정
train_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/train_tools.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
        metainfo=dataset_info,
    ))

val_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/val_tools.json',
        bbox_file=None,
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dataset_info,
    ))

test_dataloader = val_dataloader

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=48,
        out_channels=3,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)
    ),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# training schedule for 210 epochs
train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=10)
val_cfg = dict()
test_cfg = dict()

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val_tools.json')
test_evaluator = val_evaluator

# WandB 로깅 설정 추가
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)