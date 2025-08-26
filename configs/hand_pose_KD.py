# _base_ = ['../../../_base_/default_runtime.py']

max_epochs = 420
stage2_num_epochs = 30
base_lr = 4e-3

train_cfg = dict(by_epoch = True, max_epochs = max_epochs, val_interval = 10)
val_cfg = dict()
test_cfg = dict()
randomness = dict(seed = 21)

default_scope = 'mmpose'

optim_wrapper = dict(
    type = 'OptimWrapper',
    optimizer = dict(type = 'AdamW', lr = base_lr, weight_decay = 0.05),
    paramwise_cfg = dict(
        bias_decay_mult = 0,
        bypass_duplicate = True,
        norm_decay_mult = 0
    ),
    
)

param_scheduler = [
    dict(
        type = 'LinearLR',
        start_factor = 1.0e-5,
        by_epoch = False,
        begin = 0,
        end = 1000
    ),
    dict(
        type = 'CosineAnnealingLR',
        eta_min = base_lr * 0.05, # 0.0002
        begin = max_epochs // 2,
        end = max_epochs,
        T_max = max_epochs // 2,
        by_epoch = True,
        convert_to_iter_based = True
    )
]

auto_scale_lr = dict(base_batch_size = 1024)

codec = dict(
    type='SimCCLabel',
    input_size = (256, 256),
    sigma = (5.66, 5.66),
    simcc_split_ratio = 2.0,
    normalize = False,
    use_dark = False
)

# Model
model = dict(
    type = 'TopdownPoseEstimator',
    data_preprocessor = dict(
        type = 'PoseDataPreprocessor',
        mean = [123.675, 116.28, 103.53],
        std = [58.395, 57.12, 57.375],
        bgr_to_rgb = True
    ),
    backbone = dict(
        _scope_ = 'mmdet',
        type = 'CSPNeXt',
        arch = 'P5',
        expand_ratio = 0.5,
        deepen_factor = 0.67,
        widen_factor = 0.75,
        out_indices = (4, ),
        channel_attention = True,
        norm_cfg = dict(type = 'SyncBN'),
        act_cfg = dict(type = 'SiLU'),
        init_cfg = dict(
            type = 'Pretrained',
            prefix = 'backbone',
            checkpoint = 'checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
        )
    ),
    # backbone = dict(
    #     _scope_ = 'mmdet',
    #     type = 'CSPNeXt',
    #     arch='P5'
    #     act_cfg = dict(type = 'SiLU'),
    #     arch = 'P5',
    #     channel_attention = True,
    #     deepen_factor = 0.33,
    #     expand_ratio = 0.5,
    #     init_cfg = dict(
    #         checkpoint = 
    #         '',
    #         prefix = 'backbone.',
    #         type = 'Pretrained'),
    #     norm_cfg = dict(type = 'SyncBN'),
    #     out_indices = (4, ),
    #     widen_factor = 0.5
    # # ),
    head = dict(
        type = 'RTMCCHead',
        in_channels = 768,
        out_channels = 6,
        input_size = codec['input_size'],
        in_featuremap_size = tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio = codec['simcc_split_ratio'],
        final_layer_kernel_size = 7,
        gau_cfg = dict(
            hidden_dims = 256,
            s = 128,
            expansion_factor = 2,
            dropout_rate = 0.,
            drop_path = 0.,
            act_fn = 'SiLU',
            use_rel_bias = False,
            pos_enc = False
        ),
        loss = dict(
            type = 'KLDiscretLoss',
            use_target_weight = True,
            beta = 10.,
            label_softmax = True
        ),
        decoder = codec
    ),
    # head = dict(
    #     type = 'HeatmapHead',
    #     in_channels = 512,
    #     out_channels = 6,
    #     loss = dict(type = 'KeypointMSELoss', use_target_weight = True),
    #     decoder = codec
    # ),
    test_cfg = dict(flip_test = True)
)

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/hand_tool_dataset'
backend_args = dict(backend = 'local')

dataset_info = dict(
    classes = ('left_hand', 'right_hand'),
    dataset_name = 'hands_only_dataset',
    flip_indices = [0, 1, 2, 3, 4, 5],
    joint_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    keypoint_info = dict({
        0 : dict(color = [255, 0, 0], id = 0,
                 name = 'Thumb', swap = 'Thumb', type = '.'),
        1 : dict(color = [255, 165, 0], id = 1,
                 name = 'Middle', swap = 'Middle', type = '.'),
        2 : dict(color = [255, 255, 0], id = 2,
                 name = 'Index', swap = 'Index', type = '.'),
        3 : dict(color = [0, 128, 0], id = 3,
                 name = 'Ring', swap = 'Ring', type = '.'),
        4 : dict(color = [128, 0, 128], id = 4,
                 name = 'Pinky', swap = 'Pinky', type = '.'),
        5 : dict(color = [128, 0, 128], id = 5,
                 name = 'Back_of_hand', swap = 'Back_of_hand', type = '.')
    }),
    # paper_info = dict(author = 'MyTeam', title = 'Hand Keypoint Detection', year = 2025),
    sigmas = [0.025, 0.025, 0.025, 0.025, 0.025, 0.025],
    skeleton_info = dict()
)



env_cfg = dict(
    cudnn_benchmark = False,
    mp_cfg = dict(mp_start_method = 'fork', opencv_num_threads = 0),
    dist_cfg = dict(backend = 'nccl')
)

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    type = 'PoseLocalVisualizer',
    vis_backends = [dict(type = 'LocalVisBackend')],
    name = 'visualizer'
)

log_processor = dict(
    type = 'LogProcessor',
    window_size = 50,
    by_epoch = True,
    num_digits = 6
)

log_level = 'INFO'

train_pipeline = [
    dict(type = 'LoadImage', backend_args = dict(backend = 'local')),
    dict(type = 'GetBBoxCenterScale'),
    dict(direction = 'horizontal', type = 'RandomFlip'),
    # dict(type = 'RandomHalfBody'),
    dict(
        rotate_factor = 80,
        scale_factor = [0.6, 1.4],
        type = 'RandomBBoxTransform'
    ),
    dict(
        input_size = codec['input_size'],
        type = 'TopdownAffine'
    ),
    dict(type = 'mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms = [
            dict(p = 0.1, type = 'Blur'),
            dict(p = 0.1, type = 'MedianBlur'),
            dict(
                max_height = 0.4,
                max_holes = 1,
                max_width = 0.4,
                min_height = 0.2,
                min_holes = 1,
                min_width = 0.2,
                p = 1.0,
                type = 'CoarseDropout'
            ),
        ],
        type = 'Albumentation'
    ),
    dict(
        encoder = codec,
        type = 'GenerateTarget'
    ),
    dict(type = 'PackPoseInputs')
]
train_pipeline_stage2 = [
    dict(type = 'LoadImage', backend_args = backend_args),
    dict(type = 'GetBBoxCenterScale'),
    dict(direction = 'horizontal', type = 'RandomFlip'),
    dict(type = 'RandomHalfBody'),
    dict(
        rotate_factor = 60,
        scale_factor = [0.75, 1.25],
        shift_factor = 0.0,
        type = 'RandomBBoxTransform'
    ),
    dict(
        input_size = codec['input_size'],
        type = 'TopdownAffine'
    ),
    dict(type = 'mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms = [
            dict(p = 0.1, type = 'Blur'),
            dict(p = 0.1, type = 'MedianBlur'),
            dict(
                max_height = 0.4,
                max_holes = 1,
                max_width = 0.4,
                min_height = 0.2,
                min_holes = 1,
                min_width = 0.2,
                p = 0.5,
                type = 'CoarseDropout'),
        ],
        type = 'Albumentation'
    ),
    dict(
        encoder = codec,
        type = 'GenerateTarget'
    ),
    dict(type = 'PackPoseInputs')
]

train_dataloader = dict(
    batch_size = 16,
    num_workers = 1,
    persistent_workers = True,
    sampler = dict(shuffle = True, type = 'DefaultSampler'),
    dataset = dict(
        type = dataset_type,
        data_root = data_root,
        data_mode = data_mode,
        ann_file = 'annotations/train_hands.json',
        data_prefix = dict(img = 'train/'),
        pipeline = train_pipeline,
        metainfo = dataset_info
    )
)

val_pipeline = [
    dict(type = 'LoadImage', backend_args = dict(backend = 'local')),
    dict(type = 'GetBBoxCenterScale'),
    dict(type = 'TopdownAffine',  input_size = codec['input_size']),
    dict(type = 'PackPoseInputs')
]

val_dataloader = dict(
    batch_size = 16,
    num_workers = 1,
    persistent_workers = True,
    drop_last = False,
    sampler = dict(type = 'DefaultSampler', shuffle = False, round_up = False),
    dataset = dict(
        type = dataset_type,
        data_root = data_root,
        data_mode = data_mode,
        ann_file = 'annotations/val_hands.json',
        data_prefix = dict(img = 'val/'),
        metainfo = dataset_info,
        pipeline = val_pipeline,
        test_mode = True,
    )
)

test_dataloader = val_dataloader

default_hooks = dict(
    timer = dict(type = 'IterTimerHook'),
    logger = dict(interval = 50, type = 'LoggerHook'),
    param_scheduler = dict(type = 'ParamSchedulerHook'),
    checkpoint = dict(
        interval = 10,
        max_keep_ckpts = 1,
        rule = 'greater',
        save_best = 'coco/AP',
        type = 'CheckpointHook'
    ),
    sampler_seed = dict(type = 'DistSamplerSeedHook'),
    badcase = dict(
        badcase_thr = 5,
        enable = False,
        metric_type = 'loss',
        out_dir = 'badcase',
        type = 'BadCaseAnalysisHook'
    ),
    visualization = dict(
        enable = True,
        interval = 1,
        out_dir = None,
        show = False,
        type = 'PoseVisualizationHook',
        wait_time = 0.01
    )
)

custom_hooks = [
    dict(
        ema_type = 'ExpMomentumEMA',
        momentum = 0.0002,
        priority = 49,
        type = 'EMAHook',
        update_buffers = True
    ),
    dict(
        switch_epoch = 390,
        switch_pipeline = train_pipeline_stage2,
        # [
        #     dict(backend_args = dict(backend = 'local'), type = 'LoadImage'),
        #     dict(type = 'GetBBoxCenterScale'),
        #     dict(direction = 'horizontal', type = 'RandomFlip'),
        #     dict(type = 'RandomHalfBody'),
        #     dict(
        #         rotate_factor = 60,
        #         scale_factor = [0.75, 1.25],
        #         shift_factor = 0.0,
        #         type = 'RandomBBoxTransform'
        #     ),
        #     dict(input_size = (192, 256), type = 'TopdownAffine'),
        #     dict(type = 'mmdet.YOLOXHSVRandomAug'),
        #     dict(
        #         transforms = [
        #             dict(p = 0.1, type = 'Blur'),
        #             dict(p = 0.1, type = 'MedianBlur'),
        #             dict(
        #                 max_height = 0.4,
        #                 max_holes = 1,
        #                 max_width = 0.4,
        #                 min_height = 0.2,
        #                 min_holes = 1,
        #                 min_width = 0.2,
        #                 p = 0.5,
        #                 type = 'CoarseDropout'
        #             ),
        #         ],
        #         type = 'Albumentation'),
        #     dict(
        #         encoder = dict(codec),
        #         type = 'GenerateTarget'
        #     ),
        #     dict(type = 'PackPoseInputs')
        # ],
        type = 'mmdet.PipelineSwitchHook'
    ),
]

val_evaluator = dict(
    ann_file = 'data/hand_tool_dataset/annotations/val_hands.json',
    type = 'CocoMetric')

test_evaluator = [
    dict(
        ann_file='data/hand_tool_dataset/annotations/val_hands.json',
        type='CocoMetric'),
    dict(out_file_path = 'results/pred_KD.pkl', type = 'DumpResults')
]