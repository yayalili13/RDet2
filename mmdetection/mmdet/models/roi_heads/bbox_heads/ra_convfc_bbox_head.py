# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead


@HEADS.register_module()
class RAConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 loss_entropy_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.00001),
                 loss_entropy_bbox=dict(type='L1Loss', loss_weight=0.001),
                 *args,
                 **kwargs):
        super(RAConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.num_cls_nodes = 2
        self.num_reg_nodes = 2
        self.num_cls_trees = 1
        self.num_reg_trees = 1
        self.loss_entropy_cls = build_loss(loss_entropy_cls)
        self.loss_entropy_bbox = build_loss(loss_entropy_bbox)
        self.act = nn.Sigmoid() 

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.aa_out_channels = 256
        self.aa_fcs = nn.ModuleList()
        self.aa_fcs.append(
                nn.Linear(self.in_channels * self.roi_feat_area, self.aa_out_channels)) 
        self.aa_fcs.append(
                nn.Linear(self.aa_out_channels, self.aa_out_channels)) 

        self.mm_out_channels = 64
        self.mm_convs = nn.ModuleList()
        self.mm_convs.append(
                ConvModule(
                    self.in_channels, self.mm_out_channels, 3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=norm_cfg))
        self.mm_fcs = nn.ModuleList()
        self.mm_fcs.append(
                nn.Linear(self.mm_out_channels * self.roi_feat_area, self.num_cls_nodes * self.cls_last_dim)) 

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = nn.ModuleList()
            for i in range(self.num_cls_nodes):
                self.fc_cls.append(nn.Linear(self.cls_last_dim, cls_channels))
            self.aa_cls = nn.Linear(self.aa_out_channels, self.num_cls_nodes)

        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.ModuleList()
            for i in range(self.num_reg_nodes):
                self.fc_reg.append(nn.Linear(self.reg_last_dim, out_dim_reg))
            self.aa_reg = nn.Linear(self.aa_out_channels, self.num_reg_nodes)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs'),
                        dict(name='aa_fcs'),
                        dict(name='mm_fcs')
                    ]),
                 dict(
                     type='Xavier', 
                     layer='Conv2d', 
                     distribution='normal')
            ]

    def init_weights(self):
        super(RAConvFCBBoxHead, self).init_weights()
        for m in self.fc_cls:
            if m is None:
                continue
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        for m in self.fc_reg:
            if m is None:
                continue
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.aa_cls.weight, 0, 0.01)
        nn.init.constant_(self.aa_cls.bias, 0)
        nn.init.normal_(self.aa_reg.weight, 0, 0.01)
        nn.init.constant_(self.aa_reg.bias, 0)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # routing masks
        x_mask = torch.mean(x, dim=0, keepdim=True)
        for conv in self.mm_convs:
            x_mask = conv(x_mask)
        x_mask = x_mask.flatten(1)
        for fc in self.mm_fcs:
            mask_cls = 2. *  self.act(fc(x_mask))
        mask_c = torch.chunk(mask_cls, self.num_cls_nodes, dim=1)  
      
        '''
        # this is to use the separate mask for different images in one batch. 
        if self.training:
            x_mask = torch.mean(x.view(2, -1, self.in_channels, 7, 7), dim=1, keepdim=False)
            for conv in self.mm_convs:
                x_mask = conv(x_mask)
            x_mask = x_mask.flatten(1)
            for fc in self.mm_fcs:
                mask_cls = 2. *  self.act(fc(x_mask))
            mask_cls = mask_cls.unsqueeze(1).expand(-1,512,-1)
            mask_c = torch.chunk(mask_cls.reshape(-1, 2048), self.num_cls_nodes, dim=-1)  
        else:
            x_mask = torch.mean(x, dim=0, keepdim=True)
            for conv in self.mm_convs:
                x_mask = conv(x_mask)
            x_mask = x_mask.flatten(1)
            for fc in self.mm_fcs:
                mask_cls = 2. *  self.act(fc(x_mask))
            mask_c = torch.chunk(mask_cls, self.num_cls_nodes, dim=1)  
        '''
        # routing probabilities
        xa = x.flatten(1)
        for fc in self.aa_fcs:
            xa = self.relu(fc(xa))

        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = x + conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        if self.with_cls:
            cls_score_s = []
            for i, mask_i in enumerate(mask_c):
                cls_score_s.append(self.fc_cls[i](torch.mul(x_cls, mask_i)))
            cls_score = torch.cat(cls_score_s, dim=1)
            aa_cls_ = self.aa_cls(xa)
            aa_cls_ = F.softmax(aa_cls_.view((cls_score.shape[0], self.num_cls_trees, -1)), dim=2)
            aa_cls = aa_cls_.view((cls_score.shape[0], -1))
            aa_cls = aa_cls / self.num_cls_trees
        else:
            cls_score = None
            aa_cls = None
        if self.with_reg:
            bbox_pred_s = []
            for i, mask_i in enumerate(mask_c):
                bbox_pred_s.append(self.fc_reg[i](torch.mul(x_reg, mask_i)))
            bbox_pred = torch.cat(bbox_pred_s, dim=1)
            aa_reg_ = self.aa_reg(xa)
            aa_reg_ = F.softmax(aa_reg_.view((bbox_pred.shape[0], self.num_reg_trees, -1)), dim=2)
            aa_reg = aa_reg_.view((bbox_pred.shape[0], -1))
            aa_reg = aa_reg / self.num_reg_trees
        else:
            bbox_pred = None
            aa_reg = None
       
        if not self.training:
            if cls_score is not None:
                cls_score = torch.squeeze(
                        torch.bmm(aa_cls.unsqueeze(1),
                                F.softmax(cls_score.view(cls_score.shape[0], self.num_cls_nodes, -1), dim=-1))) # (N*1*K) (N*K*D)

            if bbox_pred is not None:
                bbox_pred = torch.squeeze(
                        torch.bmm(aa_reg.unsqueeze(1),
                                bbox_pred.view(bbox_pred.shape[0], self.num_reg_nodes, -1))) # (N*1*K) (N*K*D)
            return cls_score, bbox_pred
        else:
            return cls_score, bbox_pred, aa_cls, aa_reg
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'aa_cls', 'aa_reg'))
    def loss(self,
             cls_score,
             bbox_pred,
             aa_cls,
             aa_reg,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                cls_score_F = torch.squeeze(torch.bmm(aa_cls.unsqueeze(1), cls_score.view(cls_score.shape[0], self.num_cls_nodes, -1)))
                loss_ent_cls_ = self.loss_entropy_cls(
                    cls_score,
                    aa_cls.detach(),
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                loss_cls_ = self.loss_cls(
                    aa_cls,
                    cls_score,
                    labels,
                    weight=label_weights,
                    avg_factor=avg_factor)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if isinstance(loss_ent_cls_, dict):
                    losses.update(loss_ent_cls_)
                else:
                    losses['loss_ent_cls'] = loss_ent_cls_
                
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score_F, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score_F, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                bbox_pred_F = torch.squeeze(torch.bmm(aa_reg.unsqueeze(1), bbox_pred.view(bbox_pred.shape[0], self.num_reg_nodes, -1)))
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred_F = self.bbox_coder.decode(rois[:, 1:], bbox_pred_F)
                if self.reg_class_agnostic:
                    pos_bbox_pred_F = bbox_pred_F.view(
                        bbox_pred_F.size(0), 4)[pos_inds.type(torch.bool)]
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1, 4)[pos_inds.type(torch.bool)]
                    pos_bbox_pred = pos_bbox_pred.view(pos_bbox_pred.size(0),-1)
                else:
                    pos_bbox_pred_F = bbox_pred_F.view(
                        bbox_pred_F.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_reg_nodes, -1,
                        4)[pos_inds.type(torch.bool), :,
                                labels[pos_inds.type(torch.bool)], :]
                    pos_bbox_pred = pos_bbox_pred.view(pos_bbox_pred.size(0),-1)
                pos_aa_reg = aa_reg.view(
                    aa_reg.size(0), -1)[pos_inds.type(torch.bool)]
                
                losses['loss_ent_bbox'] = self.loss_entropy_bbox(
                    pos_bbox_pred,
                    pos_aa_reg.detach(),
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred_F,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_ent_bbox'] = bbox_pred[pos_inds].sum()
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        # scores = cls_score
        scores = 8 * cls_score
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

@HEADS.register_module()
class RAShared2FCBBoxHead(RAConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RAShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class RAShared4Conv1FCBBoxHead(RAConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RAShared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=1,
            num_reg_convs=0,
            num_reg_fcs=1,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
