import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmdet.core import bbox_overlaps
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
import numpy as np

from ..builder import LOSSES
from .utils import reduce_loss, weight_reduce_loss, weighted_loss


# expand onehot labels
def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(
        (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)

    return bin_labels, bin_label_weights

def ns_cross_entropy(pred,
                        aa,
                        label,
                        weight=None,
                        groups=1,
                        reduction='mean',
                        avg_factor=None,
                        class_weight=None):
    """Calculate the CrossEntropy loss with random node selection.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C * groups), C is the number
            of classes, groups is the number of nodes.
        aa (torch.Tensor): the soft routing probabilities
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    num_classes = int(pred.size(-1) / groups)
    preds = torch.chunk(pred, groups, dim=1)
    batch_size = pred.shape[0]
    num_nodes = 2
    num_trees = int(groups / num_nodes)
    
    loss_ns = []
    for i, pred_i in enumerate(preds):
        loss_i = F.cross_entropy(pred_i, label, weight=None, reduction='none')
        loss_ns.append(loss_i.view(-1,1))
    loss_ns = torch.cat(loss_ns, dim=1) 
    loss_ns = loss_ns.view((batch_size, -1, num_nodes))
    aa_ns = aa.view((batch_size, -1, num_nodes))

    # target_n = aa_ns.argmax(dim=2)
    target_n = loss_ns.argmin(dim=2)
    decay_ = torch.ones_like(loss_ns)
    inds = range(loss_ns.shape[0]) 
    for i in range(num_trees):
        rho_min = np.random.uniform(0.1,0.3)
        rho_max = np.random.uniform(0.9,1.1)
        decay_[inds, i, :]=rho_min
        decay_[inds, i, target_n[inds,i]]=rho_max
    
    loss = loss_ns * decay_ 
    loss = loss.view((batch_size, -1))
    loss = weight_reduce_loss(
            loss, weight=weight.unsqueeze(1), reduction=reduction, avg_factor=avg_factor)
    return 2 * loss / groups 


def ns_sigmoid_focal(pred,
                        aa,
                        label,
                        gamma,
                        alpha,
                        weight=None,
                        groups=1,
                        reduction='mean',
                        avg_factor=None,
                        class_weight=None):
    """Calculate the Focal Loss with node selection.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C * groups), C is the number
            of classes.
        aa (torch.Tensor): the soft routing probabilities
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    num_classes = int(pred.size(-1) / groups)
    
    preds = torch.chunk(pred, groups, dim=1)
    batch_size = pred.shape[0]
    num_nodes = 2
    num_trees = int(groups / num_nodes)
    
    loss_ns = []
    for i, pred_i in enumerate(preds):
        loss_i = _sigmoid_focal_loss(pred_i.contiguous(), label, gamma, alpha, None,'none')
        loss_i = torch.sum(loss_i, dim=-1)
        loss_ns.append(loss_i.view(-1,1))
    loss_ns = torch.cat(loss_ns, dim=1) 

    loss_ns = loss_ns.view((batch_size, -1, num_nodes))
    aa = aa.view((batch_size, -1, num_nodes))
    # target_n = loss_ns.argmin(dim=2)
    target_n = aa.argmax(dim=2)
    decay_ = torch.ones_like(loss_ns)
    inds = range(loss_ns.shape[0]) 
    for i in range(num_trees):
        rho_max = np.random.uniform(0.9,1.1)
        rho_min = np.random.uniform(0.1,0.3)
        decay_[inds, i, :]=rho_min
        decay_[inds, i, target_n[inds,i]]=rho_max
    loss = loss_ns * decay_ 
    loss = loss.view((batch_size, -1))
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return 2 * loss / groups 

def ns_l1_loss(pred, aa, target, weight=None, reduction='mean', avg_factor=None):
    """L1 loss with node selection.

    Args:
        pred (torch.Tensor): The prediction.
        aa (torch.Tensor): the soft routing probabilities
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size(0) == target.size(0) and target.numel() > 0
    groups_ = int(pred.size(1)/target.size(1))
    
    preds = torch.chunk(pred, groups_, dim=1)
    batch_size = pred.shape[0]
    num_nodes = 2
    num_trees = int(groups_ / num_nodes)
    
    loss_ns = []
    for i, pred_i in enumerate(preds):
        loss_i = torch.sum(abs(pred_i - target), dim=1)
        loss_ns.append(loss_i.view(-1,1))
    loss_ns = torch.cat(loss_ns, dim=1) 
    loss_ns = loss_ns.view((batch_size, -1, num_nodes))
    aa = aa.view((batch_size, -1, num_nodes))

    target_n = aa.argmax(dim=2)
    # target_n = loss_ns.argmin(dim=2)
    decay_ = torch.ones_like(loss_ns)
    inds = range(loss_ns.shape[0]) 
    for i in range(num_trees):
        rho_min = np.random.uniform(0.1,0.3)
        rho_max = np.random.uniform(0.9,1.1)
        decay_[inds, i, :]=rho_min
        decay_[inds, i, target_n[inds,i]]=rho_max
    
    loss = loss_ns * decay_ 
    loss = loss.view((batch_size, -1))
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return 2. * loss / groups_

# @weighted_loss
def ns_smooth_l1_loss(pred, aa, target, beta=1.0, weight=None, reduction='mean', avg_factor=None):
    """Smooth L1 loss for groups of predictions.

    Args:
        pred (torch.Tensor): The prediction.
        aa (torch.Tensor): the soft routing probabilities
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    assert pred.size(0) == target.size(0) and target.numel() > 0

    groups_ = int(pred.size(1)/target.size(1))
    preds = torch.chunk(pred, groups_, dim=1)
    batch_size = pred.shape[0]
    num_nodes = 2
    num_trees = int(groups_ / num_nodes)

    loss_ns = []
    for i, pred_i in enumerate(preds):
        diff = torch.abs(pred_i - target)
        loss_i = torch.where(diff < beta, 0.5 * diff * diff / beta,
                             diff - 0.5 * beta)
        loss_i = torch.sum(loss_i, dim=1)
        loss_ns.append(loss_i.view(-1,1))
    loss_ns = torch.cat(loss_ns, dim=1) 
    loss_ns = loss_ns.view((batch_size, -1, num_nodes))
    aa = aa.view((batch_size, -1, num_nodes))

    target_n = aa.argmax(dim=2)
    # target_n = loss_ns.argmin(dim=2)
    decay_ = torch.ones_like(loss_ns)
    inds = range(loss_ns.shape[0]) 
    for i in range(num_trees):
        rho_min = np.random.uniform(0.1,0.3)
        rho_max = np.random.uniform(0.9,1.1)
        decay_[inds, i, :]=rho_min
        decay_[inds, i, target_n[inds,i]]=rho_max
    loss = loss_ns * decay_ 
    loss = loss.view((batch_size, -1))
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return 2. * loss / groups_


@mmcv.jit(derivate=True, coderize=True)
def ns_iou_loss(pred, aa, target, weight=None, reduction='mean', avg_factor=None):
    """Smooth L1 loss for groups of predictions.

    Args:
        pred (torch.Tensor): The prediction.
        aa (torch.Tensor): the soft routing probabilities
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    eps = 1e-6
    groups_ = int(pred.size(1)/target.size(1))
    preds = torch.chunk(pred, groups_, dim=1)
    batch_size = pred.shape[0]
    num_nodes = 2
    num_trees = int(groups_ / num_nodes)

    loss_ns = []
    for i, pred_i in enumerate(preds):
        ious = bbox_overlaps(pred_i, target, is_aligned=True).clamp(min=eps)
        loss_i = -ious.log()
        loss_ns.append(loss_i.view(-1,1))
    loss_ns = torch.cat(loss_ns, dim=1) 
    loss_ns = loss_ns.view((batch_size, -1, num_nodes))
    aa = aa.view((batch_size, -1, num_nodes))

    # target_n = aa.argmax(dim=2)
    target_n = loss_ns.argmin(dim=2)
    decay_ = torch.ones_like(loss_ns)
    inds = range(loss_ns.shape[0]) 
    for i in range(num_trees):
        rho_min = np.random.uniform(0.1,0.3)
        rho_max = np.random.uniform(0.9,1.1)
        decay_[inds, i, :]=rho_min
        decay_[inds, i, target_n[inds,i]] = rho_max
   
    loss = loss_ns * decay_ 
    loss = loss.view((batch_size, -1))
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return 2. * loss / groups_

@mmcv.jit(derivate=True, coderize=True)
def ns_giou_loss(pred, aa, target, eps=1e-7, weight=None, reduction='mean', avg_factor=None):
    """giou loss for groups of predictions.

    Args:
        pred (torch.Tensor): The prediction.
        aa (torch.Tensor): the soft routing probabilities
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size(0) == target.size(0) and target.numel() > 0

    groups_ = int(pred.size(1)/target.size(1))
    preds = torch.chunk(pred, groups_, dim=1)
    batch_size = pred.shape[0]
    num_nodes = 2
    num_trees = int(groups_ / num_nodes)

    loss_ns = []
    for i, pred_i in enumerate(preds):
        gious = bbox_overlaps(pred_i, target, mode='giou', is_aligned=True, eps=eps)
        loss_i = 1 - gious
        loss_ns.append(loss_i.view(-1,1))
    loss_ns = torch.cat(loss_ns, dim=1) 
    loss_ns = loss_ns.view((batch_size, -1, num_nodes))
    aa = aa.view((batch_size, -1, num_nodes))

    # target_n = aa.argmax(dim=2)
    target_n = loss_ns.argmin(dim=2)
    decay_ = torch.ones_like(loss_ns)
    inds = range(loss_ns.shape[0]) 
    for i in range(num_trees):
        rho_min = np.random.uniform(0.1,0.3)
        rho_max = np.random.uniform(0.9,1.1)
        decay_[inds, i, :]=rho_min
        decay_[inds, i, target_n[inds,i]] = rho_max
    loss = loss_ns * decay_ 
    loss = loss.view((batch_size, -1))
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return 2. * loss / groups_


@LOSSES.register_module()
class NSL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(NSL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                aa,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The predictions from different nodes.
            aa (torch.Tensor): The routing probabilities.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * ns_l1_loss(
                pred, aa, 
                target, 
                weight=torch.unsqueeze(weight[:,0],1), 
                reduction=reduction, 
                avg_factor=avg_factor)
        return loss_bbox

@LOSSES.register_module()
class NSSmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(NSSmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                aa,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The predictions from different nodes.
            aa (torch.Tensor): The routing probabilities.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * ns_smooth_l1_loss(
            pred,
            aa,
            target,
            weight=torch.unsqueeze(weight[:,0],1),
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox

@LOSSES.register_module()
class NSIoULoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(NSIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.iter_ = 0

    def forward(self,
                pred,
                aa,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The predictions from different nodes.
            aa (torch.Tensor): The routing probabilities.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        if weight.ndim > 1:
            # weight = torch.unsqueeze(weight[:,0],1)
            weight = weight[:,0]
        loss_bbox = self.loss_weight * ns_iou_loss(
                pred, 
                aa, 
                target, 
                weight=torch.unsqueeze(weight,1), 
                reduction=reduction, 
                avg_factor=avg_factor)
        return loss_bbox

@LOSSES.register_module()
class NSCrossEntropyLoss(nn.Module):

    def __init__(self,
                 num_groups=1,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(NSCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.num_groups = num_groups
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        assert (use_sigmoid is False) or (use_mask is False)
        self.cls_criterion = ns_cross_entropy

    def forward(self,
                cls_score,
                aa_cls,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            aa_cls,
            label,
            weight,
            groups=self.num_groups,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

@LOSSES.register_module()
class NSFocalLoss(nn.Module):

    def __init__(self,
                 num_groups=1,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(NSFocalLoss, self).__init__()
        # assert (use_sigmoid is False) or (use_mask is False)
        self.num_groups = num_groups
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def forward(self,
                cls_score,
                aa_cls,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        
        if weight is not None:
            weight=torch.unsqueeze(weight,1)

        loss_cls = self.loss_weight * ns_sigmoid_focal(
            cls_score,
            aa_cls,
            label,
            gamma=self.gamma,
            alpha=self.alpha,
            weight=weight,
            groups=self.num_groups,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


@LOSSES.register_module()
class NSGIoULoss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(NSGIoULoss, self).__init__()
        self.eps = 1e-7
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                aa,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight.ndim > 1:
            # weight = torch.unsqueeze(weight[:,0],1)
            weight = weight[:,0]
        loss_bbox = self.loss_weight * ns_giou_loss(
            pred,
            aa,
            target,
            eps=self.eps,
            weight=torch.unsqueeze(weight, 1),
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
