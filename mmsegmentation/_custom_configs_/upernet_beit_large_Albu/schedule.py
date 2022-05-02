optimizer = dict(
    type='AdamW',
    lr=2e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95))
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook', cumulative_iters=2)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=5)
evaluation = dict(metric='mIoU', save_best='mIoU')
seed = 2022
work_dir = '/opt/ml/input/mmsegmentation/work_dirs/upernet_beit_large_Albu'
fp16 = dict()
gpu_ids = [0]
auto_resume = False