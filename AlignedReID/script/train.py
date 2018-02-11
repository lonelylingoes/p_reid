#-*- coding:utf-8 -*-
#===================================
# train program
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

import time
import os.path as osp
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import config.config as config
import utils.common_utils as common_utils 
import utils.model_utils as model_utils
import model.model as model 
import model.loss as loss
from utils.model_utils import transer_var_tensor


def train(train_loader, model, loss_dict, optimizer, epoch, cfg):
    '''
    one epoch train function
    args:
        train_loader:
        model: 
        loss_dict: total_loss dict
        optimizer: 
        epoch: current epoch
        cfg: config
    '''
    # switch to train mode
    model.train()

    meter_dict = dict(
        g_prec_meter = model_utils.AverageMeter(),
        g_m_meter = model_utils.AverageMeter(),
        g_dist_ap_meter = model_utils.AverageMeter(),
        g_dist_an_meter = model_utils.AverageMeter(),
        g_loss_meter = model_utils.AverageMeter(),
        l_prec_meter = model_utils.AverageMeter(),
        l_m_meter = model_utils.AverageMeter(),
        l_dist_ap_meter = model_utils.AverageMeter(),
        l_dist_an_meter = model_utils.AverageMeter(),
        l_loss_meter = model_utils.AverageMeter(),
        id_loss_meter = model_utils.AverageMeter(),
        loss_meter = model_utils.AverageMeter())

    epoch_start = time.time()
    for step, (ims, labels) in enumerate(train_loader):
        step_start = time.time()
        # change the shape of ims and labels
        ims = ims.view(-1, ims.size()[2], ims.size()[3], ims.size()[4])
        labels = labels.view(-1, )

        ims_var = Variable(transer_var_tensor(ims.float()))
        labels_t = transer_var_tensor(labels.long())
        labels_var = Variable(labels_t)

        global_feat, local_feat, logits = model(ims_var)

        g_loss, p_inds, n_inds, g_dist_ap, g_dist_an, g_dist_mat = loss.global_loss(
            loss_dict['g_tri_loss'], global_feat, labels_t,
            normalize_feature=cfg.normalize_feature)

        if cfg.l_loss_weight == 0:
            l_loss = 0
        elif cfg.local_dist_own_hard_sample:
            # Let local distance find its own hard samples.
            l_loss, l_dist_ap, l_dist_an, _ = loss.local_loss(
                loss_dict['l_tri_loss'], local_feat, None, None, labels_t,
                normalize_feature=cfg.normalize_feature)
        else:
            l_loss, l_dist_ap, l_dist_an = loss.local_loss(
                loss_dict['l_tri_loss'], local_feat, p_inds, n_inds, labels_t,
                normalize_feature=cfg.normalize_feature)

        id_loss = 0
        if cfg.id_loss_weight > 0:
            id_loss = loss_dict['id_criterion'](logits, labels_var)

        total_loss = g_loss * cfg.g_loss_weight \
                + l_loss * cfg.l_loss_weight \
                + id_loss * cfg.id_loss_weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # precision
        g_prec = (g_dist_an > g_dist_ap).data.float().mean()
        # the proportion of triplets that satisfy margin
        g_m = (g_dist_an > g_dist_ap + cfg.global_margin).data.float().mean()
        g_d_ap = g_dist_ap.data.mean()
        g_d_an = g_dist_an.data.mean()

        meter_dict['g_prec_meter'].update(g_prec)
        meter_dict['g_m_meter'].update(g_m)
        meter_dict['g_dist_ap_meter'].update(g_d_ap)
        meter_dict['g_dist_an_meter'].update(g_d_an)
        meter_dict['g_loss_meter'].update(common_utils.to_scalar(g_loss))

        if cfg.l_loss_weight > 0:
            # precision
            l_prec = (l_dist_an > l_dist_ap).data.float().mean()
            # the proportion of triplets that satisfy margin
            l_m = (l_dist_an > l_dist_ap + cfg.local_margin).data.float().mean()
            l_d_ap = l_dist_ap.data.mean()
            l_d_an = l_dist_an.data.mean()

            meter_dict['l_prec_meter'].update(l_prec)
            meter_dict['l_m_meter'].update(l_m)
            meter_dict['l_dist_ap_meter'].update(l_d_ap)
            meter_dict['l_dist_an_meter'].update(l_d_an)
            meter_dict['l_loss_meter'].update(common_utils.to_scalar(l_loss))

        if cfg.id_loss_weight > 0:
            meter_dict['id_loss_meter'].update(common_utils.to_scalar(id_loss))

        meter_dict['loss_meter'].update(common_utils.to_scalar(total_loss))

        # step log
        step_log(meter_dict, step_start, cfg, epoch, step)
    # Epoch Log 
    epoch_log(meter_dict, epoch_start, cfg, epoch)
    # tensorboar log
    tensorBoard_log(meter_dict, cfg, epoch, writer=None)
    # save ckpt
    model_utils.save_ckpt(model, optimizer, epoch + 1, cfg.ckpt_file)


def step_log(meter_dict, step_start, cfg, epoch, step):
    '''
    Log every epoch
    args:
        meter_dict: meters data
        epoch_start: start time of the epoch 
        cfg: config 
        epoch: current epoch
        step: currnet step
    '''
    if step % cfg.log_steps == 0:
        return

    time_log = '\tStep {}/epoch {}, {:.2f}s'.format(
        step, epoch + 1, time.time() - step_start, )

    if cfg.g_loss_weight > 0:
        g_log = (', gp {:.2%}, gm {:.2%}, '
                'gd_ap {:.4f}, gd_an {:.4f}, '
                'gL {:.4f}'.format(
            meter_dict['g_prec_meter'].val, meter_dict['g_m_meter'].val,
            meter_dict['g_dist_ap_meter'].val, meter_dict['g_dist_an_meter'].val,
            meter_dict['g_loss_meter'].val, ))
    else:
        g_log = ''

    if cfg.l_loss_weight > 0:
        l_log = (', lp {:.2%}, lm {:.2%}, '
                'ld_ap {:.4f}, ld_an {:.4f}, '
                'lL {:.4f}'.format(
            meter_dict['l_prec_meter'].val, meter_dict['l_m_meter'].val,
            meter_dict['l_dist_ap_meter'].val, meter_dict['l_dist_an_meter'].val,
            meter_dict['l_loss_meter'].val, ))
    else:
        l_log = ''

    if cfg.id_loss_weight > 0:
        id_log = (', idL {:.4f}'.format(meter_dict['id_loss_meter'].val))
    else:
        id_log = ''

    total_loss_log = ', total_loss {:.4f}'.format(meter_dict['loss_meter'].val)

    log = time_log + \
        g_log + l_log + id_log + \
        total_loss_log
    print(log)



def epoch_log(meter_dict, epoch_start, cfg, epoch):
    '''
    Log every epoch
    args:
        meter_dict: meters data
        epoch_start: start time of the epoch 
        cfg: config 
        epoch: current epoch
    '''
    time_log = 'epoch {}, {:.2f}s'.format(epoch + 1, time.time() - epoch_start, )

    if cfg.g_loss_weight > 0:
        g_log = (', gp {:.2%}, gm {:.2%}, '
                'gd_ap {:.4f}, gd_an {:.4f}, '
                'gL {:.4f}'.format(
            meter_dict['g_prec_meter'].avg, meter_dict['g_m_meter'].avg,
            meter_dict['g_dist_ap_meter'].avg, meter_dict['g_dist_an_meter'].avg,
            meter_dict['g_loss_meter'].avg, ))
    else:
        g_log = ''

    if cfg.l_loss_weight > 0:
        l_log = (', lp {:.2%}, lm {:.2%}, '
                'ld_ap {:.4f}, ld_an {:.4f}, '
                'lL {:.4f}'.format(
            meter_dict['l_prec_meter'].avg, meter_dict['l_m_meter.avg'],
            meter_dict['l_dist_ap_meter'].avg, meter_dict['l_dist_an_meter'].avg,
            meter_dict['l_loss_meter'].avg, ))
    else:
        l_log = ''

    if cfg.id_loss_weight > 0:
        id_log = (', idL {:.4f}'.format(meter_dict['id_loss_meter'].avg))
    else:
        id_log = ''

    total_loss_log = ', total_loss {:.4f}'.format(meter_dict['loss_meter'].avg)

    log = time_log + \
        g_log + l_log + id_log + \
        total_loss_log
    print(log)


def tensorBoard_log(meter_dict, cfg, epoch, writer=None):
    '''
    Log to TensorBoard
    args:
        meter_dict: meters data
        cfg: config 
        epoch: current epoch
        writer: the writer used to write log for tensorboard
    '''
    if not cfg.log_to_file:
        return
        
    if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
    
    writer.add_scalars(
        'total_loss',
        dict(global_loss=meter_dict['g_loss_meter'].avg,
            local_loss=meter_dict['l_loss_meter'].avg,
            id_loss=meter_dict['id_loss_meter'].avg,
            total_loss=meter_dict['loss_meter'].avg, ),
        epoch)
    writer.add_scalars(
        'tri_precision',
        dict(global_precision=meter_dict['g_prec_meter'].avg,
            local_precision=meter_dict['l_prec_meter'].avg, ),
        epoch)
    writer.add_scalars(
        'satisfy_margin',
        dict(global_satisfy_margin=meter_dict['g_m_meter'].avg,
            local_satisfy_margin=meter_dict['l_m_meter'].avg, ),
        epoch)
    writer.add_scalars(
        'global_dist',
        dict(global_dist_ap=meter_dict['g_dist_ap_meter'].avg,
            global_dist_an=meter_dict['g_dist_an_meter'].avg, ),
        epoch)
    writer.add_scalars(
        'local_dist',
        dict(local_dist_ap=meter_dict['l_dist_ap_meter'].avg,
            local_dist_an=meter_dict['l_dist_an_meter'].avg, ),
        epoch)