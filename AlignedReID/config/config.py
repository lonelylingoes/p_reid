#-*- coding:utf-8 -*-
#===================================
# config program
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

from reid_utils.common_utils import load_pickle
from reid_utils.common_utils import time_str
from reid_utils.common_utils import str2bool
from reid_utils.common_utils import tight_float_str as tfs

import numpy as np
import argparse
import os.path as osp


class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,1,2,3,),)
    parser.add_argument('--num_models', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=False)
    parser.add_argument('--train_dataset', type=str, default='market1501',
                    choices=['market1501', 'cuhk03', 'duke', 'combined'])
    parser.add_argument('--test_dataset', type=str, default='')
    parser.add_argument('--train_dataset_partitions', type=str, default='/data/DataSet/market1501/partitions.pkl')
    parser.add_argument('--test_dataset_partitions', type=str, default='')
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    # Only for training set.
    parser.add_argument('--ids_per_batch', type=int, default=32)
    parser.add_argument('--ims_per_id', type=int, default=4)

    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--normalize_feature', type=str2bool, default=False)
    parser.add_argument('--to_re_rank', type=str2bool, default=False)
    parser.add_argument('--local_dist_own_hard_sample',
                        type=str2bool, default=False)
    parser.add_argument('-gm', '--global_margin', type=float, default=0.3)
    parser.add_argument('-lm', '--local_margin', type=float, default=0.3)
    parser.add_argument('-glw', '--g_loss_weight', type=float, default=1.)
    parser.add_argument('-llw', '--l_loss_weight', type=float, default=1.)
    parser.add_argument('-idlw', '--id_loss_weight', type=float, default=0)
    parser.add_argument('-pmlw', '--pm_loss_weight', type=float, default=1.)
    parser.add_argument('-gdmlw', '--gdm_loss_weight', type=float, default=1.)
    parser.add_argument('-ldmlw', '--ldm_loss_weight', type=float, default=0.)

    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--model_weight_file', type=str, default='')

    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay_type', type=str, default='exp',
                        choices=['exp', 'staircase'])
    parser.add_argument('--exp_decay_at_epoch', type=int, default=151)
    parser.add_argument('--staircase_decay_at_epochs',
                        type=str, default='(101, 201,)')
    parser.add_argument('--staircase_decay_multiply_factor',
                        type=float, default=0.1)
    parser.add_argument('--total_epochs', type=int, default=300)

    args = parser.parse_known_args()[0]

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    ###########
    # Dataset #
    ###########
    self.workers = 2
    self.train_dataset = args.train_dataset
    if args.test_dataset == '':
      self.test_dataset = self.train_dataset
    else:
      self.test_dataset = args.test_dataset
    self.train_dataset_partitions = args.train_dataset_partitions
    if args.test_dataset_partitions == '':
      self.test_dataset_partitions = self.train_dataset_partitions
    else:
      self.test_dataset_partitions = args.test_dataset_partitions
    self.trainset_part = args.trainset_part

    # Image Processing
    # (height, width)
    self.im_resize_size = (256, 128)
    self.im_crop_size = (256, 128)
    self.keep_ratio_size = 416# the bigger side
    self.random_rotation_degree = 30
    # will scale by 1/255
    self.im_mean = [0.485, 0.456, 0.406]
    #mean=[0.485, 0.456, 0.406]
    # Whether to divide by std, set to `None` to disable.
    # Dividing is applied only when subtracting mean is applied.
    self.im_std = [0.229, 0.224, 0.225]
    
    self.ids_per_batch = args.ids_per_batch
    self.ims_per_id = args.ims_per_id

    self.test_batch_size = 32
    self.val_at_epoch = 20
    
    ###############
    # ReID Model  #
    ###############
    self.local_dist_own_hard_sample = args.local_dist_own_hard_sample

    self.normalize_feature = args.normalize_feature
    self.to_re_rank = args.to_re_rank

    self.local_conv_out_channels = 128
    self.global_margin = args.global_margin
    self.local_margin = args.local_margin

    # Identification Loss weight
    self.id_loss_weight = args.id_loss_weight

    # global loss weight
    self.g_loss_weight = args.g_loss_weight
    # local loss weight
    self.l_loss_weight = args.l_loss_weight


    ###############
    # Mutual Loss #
    ###############
    # probability mutual loss weight
    self.pm_loss_weight = args.pm_loss_weight
    # global distance mutual loss weight
    self.gdm_loss_weight = args.gdm_loss_weight
    # local distance mutual loss weight
    self.ldm_loss_weight = args.ldm_loss_weight

    self.num_models = args.num_models
    # See method `set_devices_for_ml` in `reid_utils/reid_utils.py` for
    # details.
    '''
    assert len(self.sys_device_ids) == self.num_models, \
      'You should specify device for each model.'
    '''

    # Currently one model occupying multiple GPUs is not allowed.
    if self.num_models > 1:
      for ids in self.sys_device_ids:
        assert len(ids) == 1, "When num_models > 1, one model occupying " \
                              "multiple GPUs is not allowed."
    #############
    # Training  #
    #############

    self.weight_decay = 0.0005

    # Initial learning rate
    self.base_lr = args.base_lr
    self.lr_decay_type = args.lr_decay_type
    self.exp_decay_at_epoch = args.exp_decay_at_epoch
    self.staircase_decay_at_epochs = eval(args.staircase_decay_at_epochs)
    self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
    # Number of epochs to train
    self.total_epochs = args.total_epochs

    # How often (in batches) to log. If only need to log the average
    # information for each epoch, set this to a large value, e.g. 1e10.
    self.log_steps = 1e10

    # Only test and without training.
    self.only_test = args.only_test
    self.separate_camera_set = False
    self.single_gallery_shot = False
    self.first_match_break = True
    self.resume = args.resume

    #######
    # Log #
    #######

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        '../ckpt_dir',
        '{}'.format(self.train_dataset),
        'train',
        #
        ('nf_' if self.normalize_feature else 'not_nf_') +
        ('ohs_' if self.local_dist_own_hard_sample else 'not_ohs_') +
        'gm_{}_'.format(tfs(self.global_margin)) +
        'lm_{}_'.format(tfs(self.local_margin)) +
        'glw_{}_'.format(tfs(self.g_loss_weight)) +
        'llw_{}_'.format(tfs(self.l_loss_weight)) +
        'idlw_{}_'.format(tfs(self.id_loss_weight)) +
        'pmlw_{}_'.format(tfs(self.pm_loss_weight)) +
        'gdmlw_{}_'.format(tfs(self.gdm_loss_weight)) +
        'ldmlw_{}_'.format(tfs(self.ldm_loss_weight)) +
        'lr_{}_'.format(tfs(self.base_lr)) +
        '{}_'.format(self.lr_decay_type) +
        ('decay_at_{}_'.format(self.exp_decay_at_epoch)
         if self.lr_decay_type == 'exp'
         else 'decay_at_{}_factor_{}_'.format(
          args.staircase_decay_at_epochs,
          tfs(self.staircase_decay_multiply_factor))) +
        'total_{}'.format(self.total_epochs),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
    self.model_weight_file = args.model_weight_file
