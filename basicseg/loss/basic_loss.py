import torch
import torch.nn as nn
import torch.nn.functional as F
from basicseg.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class Iou_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        iou = intersection_sum / (pred_sum + target_sum - intersection_sum + self.eps)
        if self.reduction == 'mean':
            return 1 - iou.mean()
        elif self.reduction == 'sum':
            # print(sum)
            return 1 - iou.sum()
        else:
            raise NotImplementedError('reduction type {} not implemented'.format(self.reduction))

@LOSS_REGISTRY.register()
class Dice_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * target, dim=(1,2,3))
        total_sum = torch.sum((pred + target), dim=(1,2,3))
        dice = 2 * intersection / (total_sum + self.eps)
        if self.reduction == 'mean':
            return 1 - dice.mean()
        elif self.reduction == 'sum':
            return 1 - dice.sum()
        else:
            raise NotImplementedError('reduction type {} not implemented'.format(self.reduction))

@LOSS_REGISTRY.register()
class FocalIoULoss(nn.Module):
    def __init__(self, reduction='mean', alpha=0.75, gamma=2):
        super(FocalIoULoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction=self.reduction)
    def forward(self, inputs, targets):
        [b, c, h, w] = inputs.size()

        inputs = torch.nn.Sigmoid()(inputs)
        inputs = 0.999 * (inputs - 0.5) + 0.5
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        intersection = torch.mul(inputs, targets)
        smooth = 1

        IoU = (intersection.sum() + smooth) / (inputs.sum() + targets.sum() - intersection.sum() + smooth)

        pt = torch.exp(-BCE_loss)

        F_loss = torch.mul(((1 - pt) ** self.gamma), BCE_loss)

        at = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        F_loss = (1 - IoU) * (F_loss) ** (IoU * 0.5 + 0.5)

        F_loss_map = at * F_loss
        if self.reduction == 'mean':
            return F_loss_map.mean()
        elif self.reduction == 'sum':
            return F_loss_map.sum()
        else:
            raise NotImplementedError('reduction type {} not implemented'.format(self.reduction))

# @LOSS_REGISTRY.register()
# class Focal_loss(nn.Module):
#     def __init__(self, reduction='mean', alpha=0.25, gamma=2):
#         super(Focal_loss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction=self.reduction)
#     def forward(self, preds, target):
#         bce_loss = self.bce_loss_fn(preds, target)
#         pt = torch.exp(-bce_loss)
#         focal_loss = (1-pt)**self.gamma*bce_loss
#         if self.alpha is not None:
#             focal_loss = self.alpha*focal_loss
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             raise NotImplementedError('reduction type {} not implemented'.format(self.reduction))

# @LOSS_REGISTRY.register()
# class FocalLoss(nn.Module):
#     def __init__(self, reduction='mean', alpha=0.8, gamma=2, weight=None, ignore_index=255):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.weight = weight
#         self.ignore_index = ignore_index
#         self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)
#
#     def forward(self, preds, labels):
#         if self.ignore_index is not None:
#             mask = labels != self.ignore_index
#             labels = labels[mask]
#             preds = preds[mask]
#
#         logpt = -self.bce_fn(preds, labels)
#         pt = torch.exp(logpt)
#         loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
#         return loss

@LOSS_REGISTRY.register()
class Bce_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        # reduction：默认 ‘mean’，指定应用于输出的缩减方式，另可选 ‘none’，‘sum’；
        # ‘none’：不应用缩减；‘mean’：输出的和除以输出内元素数；‘sum’：输出加和
        loss_fn = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return loss_fn(pred, target)

@LOSS_REGISTRY.register()
class L1_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        loss_fn = nn.L1Loss(reduction=self.reduction)
        return loss_fn(pred, target)