auto_scale_lr = dict(base_batch_size=1024)
backend_args = dict(backend='local')
base_lr = 0.004
codec = dict(
    heatmap_size=(
        48,
        64,
    ),
    input_size=(
        192,
        256,
    ),
    sigma=2,
    type='MSRAHeatmap')
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=390,
        switch_pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(
                rotate_factor=60,
                scale_factor=[
                    0.75,
                    1.25,
                ],
                shift_factor=0.0,
                type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                transforms=[
                    dict(p=0.1, type='Blur'),
                    dict(p=0.1, type='MedianBlur'),
                    dict(
                        max_height=0.4,
                        max_holes=1,
                        max_width=0.4,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=0.5,
                        type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    heatmap_size=(
                        48,
                        64,
                    ),
                    input_size=(
                        192,
                        256,
                    ),
                    sigma=2,
                    type='MSRAHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_mode = 'topdown'
data_root = 'data/hand_tool_dataset/'
dataset_info = dict(
    classes=(
        'Left Hand',
        'Right Hand',
    ),
    dataset_name='hands_only_dataset',
    flip_indices=[
        0,
        1,
        2,
        3,
        4,
        5,
    ],
    joint_weights=[
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    keypoint_info=dict({
        0:
        dict(color=[
            255,
            0,
            0,
        ], id=0, name='Thumb', swap='Thumb', type='.'),
        1:
        dict(
            color=[
                255,
                0,
                0,
            ], id=1, name='Middle', swap='Middle', type='.'),
        2:
        dict(color=[
            255,
            0,
            0,
        ], id=2, name='Index', swap='Index', type='.'),
        3:
        dict(color=[
            255,
            0,
            0,
        ], id=3, name='Ring', swap='Ring', type='.'),
        4:
        dict(color=[
            255,
            0,
            0,
        ], id=4, name='Pinky', swap='Pinky', type='.'),
        5:
        dict(
            color=[
                255,
                0,
                0,
            ],
            id=5,
            name='Back_of_hand',
            swap='Back_of_hand',
            type='.')
    }),
    paper_info=dict(
        author='MyTeam', title='Hand Keypoint Detection', year=2025),
    sigmas=[
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
    ],
    skeleton_info=dict())
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=1,
        rule='greater',
        save_best='coco/AP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        enable=True,
        interval=1,
        out_dir=None,
        show=False,
        type='PoseVisualizationHook',
        wait_time=0.01))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'work_dirs/hand_pose/best_coco_AP_epoch_210.pth'
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
max_epochs = 420
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.33,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=0.5),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            heatmap_size=(
                48,
                64,
            ),
            input_size=(
                192,
                256,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        in_channels=512,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        out_channels=6,
        type='HeatmapHead'),
    test_cfg=dict(flip_test=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.0),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=210,
        begin=210,
        by_epoch=True,
        convert_to_iter_based=True,
        end=420,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=21)
resume = False
stage2_num_epochs = 30
test_cfg = dict()
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/val_hands.json',
        data_mode='topdown',
        data_prefix=dict(img='val/'),
        data_root='data/hand_tool_dataset/',
        metainfo=dict(
            classes=(
                'Left Hand',
                'Right Hand',
            ),
            dataset_name='hands_only_dataset',
            flip_indices=[
                0,
                1,
                2,
                3,
                4,
                5,
            ],
            joint_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=0,
                    name='Thumb',
                    swap='Thumb',
                    type='.'),
                1:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=1,
                    name='Middle',
                    swap='Middle',
                    type='.'),
                2:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=2,
                    name='Index',
                    swap='Index',
                    type='.'),
                3:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=3,
                    name='Ring',
                    swap='Ring',
                    type='.'),
                4:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=4,
                    name='Pinky',
                    swap='Pinky',
                    type='.'),
                5:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=5,
                    name='Back_of_hand',
                    swap='Back_of_hand',
                    type='.')
            }),
            paper_info=dict(
                author='MyTeam', title='Hand Keypoint Detection', year=2025),
            sigmas=[
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
            ],
            skeleton_info=dict()),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        ann_file='data/hand_tool_dataset/annotations/val_hands.json',
        type='CocoMetric'),
    dict(out_file_path='results/pred.pkl', type='DumpResults'),
]
train_cfg = dict(by_epoch=True, max_epochs=420, val_interval=10)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/train_hands.json',
        data_mode='topdown',
        data_prefix=dict(img='train/'),
        data_root='data/hand_tool_dataset/',
        metainfo=dict(
            classes=(
                'Left Hand',
                'Right Hand',
            ),
            dataset_name='hands_only_dataset',
            flip_indices=[
                0,
                1,
                2,
                3,
                4,
                5,
            ],
            joint_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=0,
                    name='Thumb',
                    swap='Thumb',
                    type='.'),
                1:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=1,
                    name='Middle',
                    swap='Middle',
                    type='.'),
                2:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=2,
                    name='Index',
                    swap='Index',
                    type='.'),
                3:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=3,
                    name='Ring',
                    swap='Ring',
                    type='.'),
                4:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=4,
                    name='Pinky',
                    swap='Pinky',
                    type='.'),
                5:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=5,
                    name='Back_of_hand',
                    swap='Back_of_hand',
                    type='.')
            }),
            paper_info=dict(
                author='MyTeam', title='Hand Keypoint Detection', year=2025),
            sigmas=[
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
            ],
            skeleton_info=dict()),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(
                rotate_factor=80,
                scale_factor=[
                    0.6,
                    1.4,
                ],
                type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                transforms=[
                    dict(p=0.1, type='Blur'),
                    dict(p=0.1, type='MedianBlur'),
                    dict(
                        max_height=0.4,
                        max_holes=1,
                        max_width=0.4,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=1.0,
                        type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    heatmap_size=(
                        48,
                        64,
                    ),
                    input_size=(
                        192,
                        256,
                    ),
                    sigma=2,
                    type='MSRAHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=80,
        scale_factor=[
            0.6,
            1.4,
        ],
        type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=1.0,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            heatmap_size=(
                48,
                64,
            ),
            input_size=(
                192,
                256,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=60,
        scale_factor=[
            0.75,
            1.25,
        ],
        shift_factor=0.0,
        type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=0.5,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            heatmap_size=(
                48,
                64,
            ),
            input_size=(
                192,
                256,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/val_hands.json',
        data_mode='topdown',
        data_prefix=dict(img='val/'),
        data_root='data/hand_tool_dataset/',
        metainfo=dict(
            classes=(
                'Left Hand',
                'Right Hand',
            ),
            dataset_name='hands_only_dataset',
            flip_indices=[
                0,
                1,
                2,
                3,
                4,
                5,
            ],
            joint_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=0,
                    name='Thumb',
                    swap='Thumb',
                    type='.'),
                1:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=1,
                    name='Middle',
                    swap='Middle',
                    type='.'),
                2:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=2,
                    name='Index',
                    swap='Index',
                    type='.'),
                3:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=3,
                    name='Ring',
                    swap='Ring',
                    type='.'),
                4:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=4,
                    name='Pinky',
                    swap='Pinky',
                    type='.'),
                5:
                dict(
                    color=[
                        255,
                        0,
                        0,
                    ],
                    id=5,
                    name='Back_of_hand',
                    swap='Back_of_hand',
                    type='.')
            }),
            paper_info=dict(
                author='MyTeam', title='Hand Keypoint Detection', year=2025),
            sigmas=[
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
            ],
            skeleton_info=dict()),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/hand_tool_dataset/annotations/val_hands.json',
    type='CocoMetric')
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs\\hand_pose'
