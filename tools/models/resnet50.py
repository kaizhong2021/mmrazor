# Copyright (c) OpenMMLab. All rights reserved.
# model settings
_model = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

model = dict(
    type='FXModelWrapper',
    model=_model,
    customed_skipped_method=['mmcls.models.ClsHead._get_predictions', 'mmcls.models.ClsHead._get_loss']
)
