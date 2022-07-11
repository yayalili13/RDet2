_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        type='RARoIHead',
        bbox_head=dict(
            type='RAShared4Conv1FCBBoxHead',
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='NFCrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='IoULoss', loss_weight=10.0),
            loss_entropy_cls=dict(type='NSCrossEntropyLoss', num_groups=2, use_sigmoid=False, loss_weight=1.0),
            loss_entropy_bbox=dict(type='NSIoULoss', loss_weight=10.0))),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

