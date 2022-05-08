'''
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
runner = dict(type='EpochBasedRunner', max_epochs=70)
checkpoint_config = dict(interval=5)
evaluation = dict(metric='mIoU', save_best='mIoU')
seed = 2022
work_dir = '/opt/ml/input/mmsegmentation/work_dirs/hr_Kwon'
fp16 = dict()
gpu_ids = [0]
auto_resume = False
'''

lr = 1e-4 /2  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=7e-6)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=5)
evaluation = dict(metric='mIoU', save_best='mIoU')
seed = 2022
work_dir = '/opt/ml/input/mmsegmentation/work_dir/hr_city_scape'