#-*- coding:utf-8 -*-
#===================================
# extract feature model
#===================================
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from resnet import resnet50, resnet101


class FirstStageModel(nn.Module):
  '''the first stage model for classification.
  '''
  def __init__(self, 
              num_classes,
              pretrained = True,
              base_model_dir='../ckpt_dir'):
    super(FirstStageModel, self).__init__()
    self.base = resnet101(base_model_dir, pretrained)
    planes = self.base[1]
    self.fc = nn.Linear(planes, num_classes)
    init.normal(self.fc.weight, std=0.001)
    init.constant(self.fc.bias, 0)

  def forward(self, x):
    """
    Returns:
      global_feat: shape [N, C]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    # shape [N, C, 1]
    feat = F.max_pool2d(feat, feat.size()[2:])
    # shape [N, C]
    feat = feat.view(feat.size(0), -1)
    # shape [N, num_classes]
    logist = self.fc(feat)
    return logist



class Model(nn.Module):
  '''the model with rank loss
  '''
  def __init__(self, stage1_model):
    '''
    args:
      stage1_model: the first stage model
    '''
    super(Model, self).__init__()
    self.base = stage1_model.base
    planes = self.base[1]
    self.fc = nn.Linear(planes, planes)

  def forward(self, x):
    """
    Returns:
      global_feat: shape [N, C]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    # shape [N, C, 1]
    feat = F.max_pool2d(feat, feat.size()[2:])
    # shape [N, C]
    feat = feat.view(feat.size(0), -1)
    # shape [N, c]
    feat = self.fc(feat)
    # shape [N, c], L2 normalize
    feat = F.normal(feat, p=2, dim=0)

    return feat