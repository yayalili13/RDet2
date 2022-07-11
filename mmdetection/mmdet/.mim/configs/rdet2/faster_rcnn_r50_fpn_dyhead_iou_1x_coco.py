_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=2)
    ],
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='IoULoss', loss_weight=10.0))))
