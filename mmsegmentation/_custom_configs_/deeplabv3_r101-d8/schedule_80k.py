# # optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# # runtime settings
# runner = dict(type='IterBasedRunner', max_iters=80000)
# checkpoint_config = dict(by_epoch=False, interval=8000)
# evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
# seed = 2022

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
evalutation = dict(metric='mIoU')
seed = 2022
work_dir = '/opt/ml/input/mmsegmentation/work_dirs/deeplabv3_r50-d8'