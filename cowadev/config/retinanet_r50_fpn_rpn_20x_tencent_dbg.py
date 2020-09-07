_base_ = [
    './model/retinanet_r50_fpn_rpn.py',
    './dataset/tencent_rpn_dbg.py',
    './schedule/schedule_20x.py',
    './runtime/runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(interval=20)
