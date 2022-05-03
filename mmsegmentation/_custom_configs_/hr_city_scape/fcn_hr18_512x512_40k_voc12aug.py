_base_ = [
    '../../configs/_base_/models/fcn_hr18.py', './dataset.py'
    '../../configs/_base_/default_runtime.py', '../../configs/_base_/schedules/schedule_40k.py'
]
model = dict(decode_head=dict(num_classes=21))
