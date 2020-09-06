_base_ = [
    './model/retinanet_r50_fpn_rpn.py',
    './dataset/tencent_rpn.py',
    './schedule/schedule_2x.py',
    './runtime/runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
