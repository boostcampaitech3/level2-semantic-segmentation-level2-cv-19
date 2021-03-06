log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='WandbLoggerHook',
            interval=1000,
            init_kwargs=dict(
                project='Semantic Segmentation',
                entity='next_level',
                name='hrnet_aug'))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/ml/input/mmsegmentation/hr_city_scape/epoch_40.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True