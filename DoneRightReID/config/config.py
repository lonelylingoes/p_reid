#-*- coding:utf-8 -*-
#===================================
# config program
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.common_utils import load_pickle
from utils.common_utils import time_str
from utils.common_utils import str2bool
from utils.common_utils import tight_float_str as tfs

import numpy as np
import argparse
import os.path as osp


class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=((0,),))
    parser.add_argument('--set_seed', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='market1501',
                    choices=['market1501', 'cuhk03', 'duke', 'combined'])
    parser.add_argument('--dataset_partitions', type=str, default='/data/DataSet/market1501/partitions.pkl')
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    # Only for training set.
    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--to_re_rank', type=str2bool, default=False)
    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='')

    args = parser.parse_known_args()[0]


    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    # Dataset #
    self.workers = 2
    self.dataset = args.dataset
    self.dataset_partitions = args.dataset_partitions
    self.trainset_part = args.trainset_part

    # will scale by 1/255
    self.im_mean = [0.485, 0.456, 0.406]
    #mean=[0.485, 0.456, 0.406]
    # Whether to divide by std, set to `None` to disable.
    # Dividing is applied only when subtracting mean is applied.
    self.im_std = [0.229, 0.224, 0.225]
    
    self.resume = args.resume

    # first stage
    self.first_resize_size = (256, 256)# (height, width)
    self.first_crop_size = (224, 224)
    self.first_stage_batch_size = 128
    self.first_stage_base_lr = 10e-2
    self.first_stage_final_lr = 10e-4
    self.momentum=0.9
    self.weight_decay = 5e-5

    # second stage
    self.second_resize_size = 416# the bigger side
    self.second_stage_batch = 64
    self.second_stage_base_lr = 10e-3
    self.half_every_iterations = 512
    self.sample_numbers = 5000
    self.undates = 16
    self.iterations = 4096
    self.margin = 0.1

    # Only test and without training.
    self.only_test = args.only_test
    self.test_batch_size = 64
    self.to_re_rank = args.to_re_rank

    self.separate_camera_set = False
    self.single_gallery_shot = False
    self.first_match_break = True

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        '../ckpt_dir',
        '{}'.format(self.dataset),
        'train',
        'total_{}'.format(self.iterations),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
