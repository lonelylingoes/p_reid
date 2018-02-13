#-*- coding:utf-8 -*-
#===================================
# utils for model
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
import sys
import os
import os.path as osp
import cPickle as pickle
import gc
import numpy as np
from scipy import io
import datetime
import time
from contextlib import contextmanager
import utils.common_utils as common_utils 

import torch
from torch.autograd import Variable



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
       self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / (self.count + 1e-20)



def load_ckpt(model, optimizer, ckpt_file, load_to_cpu=False, verbose=True):
    """
    Load state_dict's of modules and optimizers from file.
    Args:
        model: torch.nn.Module
        optimizer:torch.nn.optimizer 
        ckpt_file: The file path.
        load_to_cpu: Boolean. Whether to transform tensors in modules/optimizers 
        to cpu type.
    """
    map_location = (lambda storage, loc: storage) if load_to_cpu else None
    checkpoint = torch.load(ckpt_file, map_location=map_location)
    model.load_state_dict(checkpoint['state_dicts'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if verbose:
        print('Resume from ckpt {} at epoch {}'.format(ckpt_file, checkpoint['epoch']))
    return checkpoint['epoch']


def save_ckpt(model, optimizer, epoch, ckpt_file):
    """
    Save state_dict's of modules/optimizers to file. 
    Args:
        model: torch.nn.Module
        optimizer:torch.nn.optimizer 
        epoch: the current epoch number
        ckpt_file: The file path.
    Note:
        torch.save() reserves device type and id of tensors to save, so when 
        loading ckpt, you have to inform torch.load() to load these tensors to 
        cpu or your desired gpu, if you change devices.
    """
    ckpt = dict(state_dicts=model.state_dict(),
                optimizer = optimizer.state_dict(),
                epoch=epoch)
    common_utils.may_make_dir(osp.dirname(osp.abspath(ckpt_file)))
    torch.save(ckpt, ckpt_file)


def adjust_lr_exp(optimizer, base_lr, epoch, total_epoch, start_decay_at_epoch):
    """
    Decay exponentially in the later phase of training. All parameters in the 
    optimizer share the same learning rate.
    
    Args:
        optimizer: a pytorch `Optimizer` object
        base_lr: starting learning rate
        epoch: current epoch, ep >= 1
        total_epoch: total number of epochs to train
        start_decay_at_epoch: start decaying at the BEGINNING of this epoch
    
    Example:
        base_lr = 2e-4
        total_ep = 300
        start_decay_at_ep = 201
        It means the learning rate starts at 2e-4 and begins decaying after 200 
        epochs. And training stops after 300 epochs.
    
    NOTE: 
        It is meant to be called at the BEGINNING of an epoch.
    """
    assert epoch >= 1, "Current epoch number should be >= 1"

    if epoch < start_decay_at_epoch:
        return

    for g in optimizer.param_groups:
        g['lr'] = (base_lr * (0.001 ** (float(epoch + 1 - start_decay_at_epoch)
                                        / (total_epoch + 1 - start_decay_at_epoch))))
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))



def adjust_lr_staircase(optimizer, base_lr, epoch, decay_at_epochs, factor):
    """
    Multiplied by a factor at the BEGINNING of specified epochs. All 
    parameters in the optimizer share the same learning rate.
    
    Args:
        optimizer: a pytorch `Optimizer` object
        base_lr: starting learning rate
        epoch: current epoch, epoch >= 1
        decay_at_epochs: a list or tuple; learning rate is multiplied by a factor 
        at the BEGINNING of these epochs
        factor: a number in range (0, 1)
    
    Example:
        base_lr = 1e-3
        decay_at_epochs = [51, 101]
        factor = 0.1
        It means the learning rate starts at 1e-3 and is multiplied by 0.1 at the 
        BEGINNING of the 51'st epoch, and then further multiplied by 0.1 at the 
        BEGINNING of the 101'st epoch, then stays unchanged till the end of 
        training.
    
    NOTE: 
        It is meant to be called at the BEGINNING of an epoch.
    """
    assert epoch >= 1, "Current epoch number should be >= 1"

    if epoch not in decay_at_epochs:
        return

    ind = common_utils.find_index(decay_at_epochs, epoch)
    for g in optimizer.param_groups:
        g['lr'] = base_lr * factor ** (ind + 1)
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


def transer_var_tensor(var_or_tensor, device_id = 0):
    '''
    Return a copy of the input Variable or Tensor on specified device.
    '''
    return var_or_tensor.cpu() if device_id == -1 \
        else var_or_tensor.cuda(device_id)


def load_test_model(model, cfg):
    '''
    load the param from train model for test model
    args:
        model: the init model 
        cfg: Config object
    return:
        model: the model loaded the param
    '''
    map_location = (lambda storage, loc: storage)
    if cfg.model_weight_file != '':
        src_state_dict = torch.load(cfg.model_weight_file, map_location=map_location)
    else:
        checkpoint = torch.load(cfg.ckpt_file, map_location=map_location)
        src_state_dict = checkpoint['state_dicts']
        
    # del the unused layer when test
    dest_state_dict = model.state_dict()
    for name, param in src_state_dict.items():
        if name not in dest_state_dict:
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            dest_state_dict[name].copy_(param)
        except Exception, msg:
            print("Warning: Error occurs when copying '{}': {}".format(name, str(msg)))

    return model