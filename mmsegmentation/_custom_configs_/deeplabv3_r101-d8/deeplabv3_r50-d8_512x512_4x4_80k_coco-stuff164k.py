_base_ = [
    './deeplabv3_r50-d8.py',
    './coco-stuff164k.py',
    './default_runtime.py',
    './schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=171), auxiliary_head=dict(num_classes=171))