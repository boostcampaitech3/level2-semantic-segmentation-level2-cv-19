_base_ = [
    './fcn_hr18.py', './dataset.py',
    
    '/opt/ml/input/code/level2-semantic-segmentation-level2-cv-19/mmsegmentation/_custom_configs_/hr_city_scape/runtime.py',
    '/opt/ml/input/code/level2-semantic-segmentation-level2-cv-19/mmsegmentation/_custom_configs_/hr_city_scape/schedule.py'
]
model = dict(decode_head=dict(num_classes=21))
