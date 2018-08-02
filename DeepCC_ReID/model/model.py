#-*- coding:utf-8 -*-
#===================================
# extract feature model
#===================================
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from resnet import resnet50


class Model(nn.Module):
  def __init__(self, 
              local_conv_out_channels=128, 
              pretrained = True,
              base_model_dir='../ckpt_dir'):
    super(Model, self).__init__()
    self.base = resnet50(base_model_dir, pretrained)
    planes = 2048

    self.fc1 = nn.Linear(planes, 1024)
    init.normal_(self.fc1.weight, std=0.001)
    init.constant_(self.fc1.bias, 0)
    self.fc1_bn = nn.BatchNorm2d(1024)
    self.fc2_relu = nn.ReLU(inplace=True)

    self.fc2 = nn.Linear(1024, 128)
    init.normal_(self.fc2.weight, std=0.001)
    init.constant_(self.fc2.bias, 0)

    #self.global_conv = nn.Conv2d(planes, 128, 1)

    # local 
    self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
    self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
    self.local_relu = nn.ReLU(inplace=True)


  def forward(self, x):
    """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    # shape [N, C, 1]
    global_feat = F.avg_pool2d(feat, feat.size()[2:])
    # shape [N, C]
    global_feat = global_feat.view(global_feat.size(0), -1)

    # shape [N, 1024]
    global_feat = self.fc1_relu(self.fc_bn(self.fc1(global_feat)))
    # shape [N, 128]
    global_feat = self.fc2(global_feat) 
    
    # global_feat = self.global_conv(global_feat)

    # shape [N, C, H, 1]
    local_feat = torch.mean(feat, -1, keepdim=True)
    local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
    # shape [N, H, c]
    local_feat = local_feat.squeeze(-1).permute(0, 2, 1)


    return global_feat, local_feat
