_base_ = [
    'mmseg::_base_/datasets/cityscapes.py',
    'mmseg::_base_/schedules/schedule_80k.py',
    'mmseg::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth'  # noqa: E501
teacher_cfg_path = 'mmseg::segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py'  # noqa: E501
student_cfg_path = 'mmseg::deeplabv3plus/deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.py'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        distill_losses=dict(
            loss_cwd=dict(type='ChannelWiseDivergence', tau=1, loss_weight=5)),
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg')),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg')),
        loss_forward_mappings=dict(
            loss_cwd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
