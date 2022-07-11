import torch
import torch.nn as nn
import torch.nn.functional as F
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

def nf_cross_entropy(aa, pred, label, weight=None, reduction='mean', avg_factor=None):
    """Warpper of split l2 loss."""
    eps = 1e-12
    num_nodes = aa.shape[1]
    num_nodes_per_tree = 2
    num_trees = int(num_nodes / num_nodes_per_tree) 
    batch_size = pred.shape[0]
    num_classes = int(pred.shape[1] / num_nodes)

    label_F = label.unsqueeze(1).repeat(1, num_trees)
    label_F = label_F.view(-1)
    pred_F = torch.squeeze(torch.bmm(aa.view((batch_size * num_trees, -1, num_nodes_per_tree)), 
            F.softmax(pred.view(batch_size * num_trees, num_nodes_per_tree, -1),dim=2)))
    loss_F = F.nll_loss((pred_F+eps).log(), label_F, weight=None, reduction='none')
    loss_F = torch.sum(loss_F.view(batch_size, -1), dim=1) / num_trees 
    
    loss = weight_reduce_loss(loss_F, weight, reduction=reduction, avg_factor=avg_factor)
    return loss

def nf_sigmoid_focal(aa, pred, label, gamma, alpha, weight=None, reduction='mean', avg_factor=None, num_class=80):
    """Warpper of split l2 loss."""
    eps = 1e-12
    num_nodes = aa.shape[1]
    num_nodes_per_tree = 2
    num_trees = int(num_nodes / num_nodes_per_tree) 
    batch_size = pred.shape[0]
    num_classes = int(pred.shape[1] / num_nodes)

    pred_F = torch.squeeze(torch.bmm(aa.view(batch_size, -1, num_nodes), pred.view(pred.shape[0], num_nodes, -1)))
    if pred_F.dim() == 1:
        pred_F = pred_F.unsqueeze(1)

    loss_F = _sigmoid_focal_loss(pred_F.contiguous(), label, gamma, alpha, None,'none')
    loss_F = torch.sum(loss_F, dim=-1)
    # loss_F = _sigmoid_focal_loss(pred_F.contiguous(), label, gamma=2.0, alpha=0.25, weight=None, reduction='none')
    loss = weight_reduce_loss(loss_F, weight, reduction=reduction, avg_factor=avg_factor)
    return loss

@LOSSES.register_module()
class NFCrossEntropyLoss(nn.Module):
    """MixLoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, use_sigmoid=False, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        if self.use_sigmoid:
            self.cls_criterion = nf_binary_cross_entropy
        else:
            self.cls_criterion = nf_cross_entropy

    def forward(self, aa, pred, label, weight=None, avg_factor=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = self.loss_weight * self.cls_criterion(
                aa, 
                pred, 
                label, 
                weight=weight, 
                reduction=self.reduction, 
                avg_factor=avg_factor)
        return loss


@LOSSES.register_module()
class NFFocalLoss(nn.Module):
    """MixLoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, 
            gamma=2.0,
            alpha=0.25,
            reduction='mean', 
            loss_weight=1.0):
        super(NFFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, aa, pred, label, weight=None, avg_factor=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = self.loss_weight * nf_sigmoid_focal(aa, pred, label, 
            gamma=self.gamma,
            alpha=self.alpha,
            weight=weight, 
            reduction=self.reduction, 
            avg_factor=avg_factor)
        return loss
