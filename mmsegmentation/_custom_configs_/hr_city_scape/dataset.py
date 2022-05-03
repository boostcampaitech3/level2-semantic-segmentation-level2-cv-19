dataset_type = 'CustomDataset'
data_root = '/opt/ml/input/'
classes = ('Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
           'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
           'Clothing')
palette = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0],
           [64, 0, 128], [64, 0, 192], [192, 128, 64], [192, 192, 128],
           [64, 64, 128], [128, 0, 192]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
albu_transform = [
    dict(type='VerticalFlip', p=0.15),
    dict(type='HorizontalFlip', p=0.3),
    dict(type='OneOf', transforms=[
            dict(type='GaussNoise', p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='Blur', p=1.0)
        ], p=0.1),
    dict(type='OneOf', transforms=[
            dict(type='RandomGamma', p=1.0),
            dict(type='HueSaturationValue', p=1.0),
            dict(type='ChannelDropout', p=1.0),
            dict(type='ChannelShuffle', p=1.0),
            dict(type='RGBShift', p=1.0),
        ], p=0.1),
    dict(type='OneOf', transforms=[
        dict(type='ShiftScaleRotate', p=1.0),
        dict(type='RandomRotate90', p=1.0)
    ], p=0.1)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(
        type='Albu',
        transforms=albu_transform,
        keymap=dict(img='image', gt_semantic_seg='mask'),
        update_pad_shape=True
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5],
        flip=False,
        # flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + 'mmseg/images/train',
        ann_dir=data_root + 'mmseg/annotations/train',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette),
    val=dict(
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + 'mmseg/images/val',
        ann_dir=data_root + 'mmseg/annotations/val',
        pipeline=val_pipeline,
        classes=classes,
        palette=palette),
    test=dict(
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + 'mmseg/test',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette))