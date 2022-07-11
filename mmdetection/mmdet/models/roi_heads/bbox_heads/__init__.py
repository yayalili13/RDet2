# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .dii_head import DIIHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .sabl_head import SABLHead
from .scnet_bbox_head import SCNetBBoxHead
# added by Yali
from .ra_convfc_bbox_head_dyrelu import (RAConvFCBBoxHead, RAShared2FCBBoxHead,
                               RAShared4Conv1FCBBoxHead)


__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead',
    'SCNetBBoxHead',
    'RAConvFCBBoxHead', 'RAShared2FCBBoxHead', 'RAShared4Conv1FCBBoxHead'
]
