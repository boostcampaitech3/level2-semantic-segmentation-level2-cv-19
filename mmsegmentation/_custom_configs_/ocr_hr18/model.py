'''
_base_ = './fcn_hr18_512x512_40k_voc12aug.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))
'''
_base_ = [
    './ocrnet_hr18.py', './dataset.py',
    
    '/opt/ml/input/code/level2-semantic-segmentation-level2-cv-19/mmsegmentation/_custom_configs_/ocr_hr18/runtime.py',
    '/opt/ml/input/code/level2-semantic-segmentation-level2-cv-19/mmsegmentation/_custom_configs_/ocr_hr18/schedule.py'
]
#model = dict(decode_head=dict(num_classes=21))
