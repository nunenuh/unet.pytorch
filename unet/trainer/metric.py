import torch
import torch.nn as nn
import torch.nn.functional as F

# inspiration from
# https://github.com/pytorch/pytorch/issues/1249
# github.com/jeffwen/road_building_extraction/blob/master/src/utils/core.py
# and other source
def dice_coeff(input, target, smooth=1.):
    input_flat = input.view(-1)
    target_flat = target.view(-1)

    intersection = (input_flat * target_flat).sum()
    return (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)


def dice_loss(input, target, smooth=1.0):
    dcoeff = dice_coeff(input, target, smooth=smooth)
    return 1-dcoeff

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        dcoeff = dice_coeff(input, target, smooth=self.smooth)
        return 1 - dcoeff

# inspiration from
# github.com/jeffwen/road_building_extraction/blob/master/src/utils/core.py
# and other source
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, input, target):
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        bce_loss = self.bce_loss(input_flat, target_flat).double()
        dice_loss = self.dice_loss(input, target)
        return bce_loss + dice_loss
    