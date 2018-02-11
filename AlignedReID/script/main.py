#-*- coding:utf-8 -*-
#===================================
# main program
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import config.config as config
import utils.common_utils as common_utils 
import utils.model_utils as model_utils
from data_set.data_set import ReIdDataSet
from model.model import Model
from train import train
from test import test
from model.loss import TripletLoss
from torch.utils.data import DataLoader
import torchvision.transforms as transforms



def main():
    # get the config
    cfg = config.Config()

    # set cpu or gpus which will be used
    #common_utils.set_devices(cfg.sys_device_ids)

    # set seed for all possibale moudel
    if cfg.seed is not None:
        common_utils.set_seed(cfg.seed)

    # logs to both console and file.
    if cfg.log_to_file:
        common_utils.Logger(cfg.stdout_file, 'stdout', False)
        common_utils.Logger(cfg.stderr_file, 'stderr', False)

    # Dump the configurations to log.
    import pprint
    print('-' * 60)
    print('cfg.__dict__')
    pprint.pprint(cfg.__dict__)
    print('-' * 60)
    
    # create train data set
    train_transform = transforms.Compose(
                        [transforms.RandomHorizontalFlip(),
                        transforms.RandomResizedCrop(cfg.im_crop_size),
                        transforms.ToTensor(),
                        # the object of normalize should be tensor,
                        # so totensor() should called before normalize()  
                        transforms.Normalize(mean=cfg.im_mean, std=cfg.im_std)]
                        )   
    train_dataset = ReIdDataSet('~/Dataset/market1501/partitions.pkl',
                                'train',
                                train_transform,
                                cfg.ids_per_batch,
                                cfg.ims_per_id)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.ids_per_batch,
        num_workers=cfg.workers, pin_memory=True)
    # create test data set
    val_transform = transforms.Compose(
                        [transforms.Resize(cfg.im_resize_size),
                        transforms.CenterCrop(cfg.im_crop_size),
                        transforms.ToTensor(),
                        # the object of normalize should be tensor,
                        # so totensor() should called before normalize() 
                        transforms.Normalize(mean=cfg.im_mean, std=cfg.im_std)]
                        )
    val_dataset = ReIdDataSet('~/Dataset/market1501/partitions.pkl',
                                'val',
                                val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.test_batch_size,
        num_workers=cfg.workers, pin_memory=True)


    # create models
    model = Model(local_conv_out_channels=128, 
                  num_classes=len(train_dataset.ids2labels))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()


    # define loss
    loss_dict =dict(id_criterion = nn.CrossEntropyLoss().cuda() \
                    if torch.cuda.is_available() else nn.CrossEntropyLoss(),
        g_tri_loss = TripletLoss(margin=cfg.global_margin),
        l_tri_loss = TripletLoss(margin=cfg.local_margin),
        g_l_tri_loss = TripletLoss(margin=cfg.local_margin + cfg.global_margin))


    # [NOTE] 
    # If you need to move a model to GPU via .cuda(), 
    # please do so before constructing optimizers for it. 
    # Parameters of a model after .cuda() will be different objects with those before the call.
    # In general, you should make sure that optimized parameters live in consistent locations 
    #   when optimizers are constructed and used.
    optimizer = optim.Adam(model.parameters(),
                            lr=cfg.base_lr,
                            weight_decay=cfg.weight_decay)


    # optionally resume from a checkpoint
    if cfg.resume:
        resume_epoch = model_utils.load_ckpt(model, optimizer, cfg.ckpt_file)

    # [NOTE]
    # It enables benchmark mode in cudnn.
    # If your input size is changing a lot, then it might hurt runtime
    # if not, it should be much faster.
    cudnn.benchmark = True

    start_epoch = resume_epoch if cfg.resume else 0
    for epoch in range(start_epoch, cfg.total_epochs):
        # Adjust Learning Rate
        if cfg.lr_decay_type == 'exp':
            model_utils.adjust_lr_exp(
                optimizer,
                cfg.base_lr,
                epoch + 1,
                cfg.total_epochs,
                cfg.exp_decay_at_epoch)
        else:
            model_utils.adjust_lr_staircase(
                optimizer,
                cfg.base_lr,
                epoch + 1,
                cfg.staircase_decay_at_epochs,
                cfg.staircase_decay_multiply_factor)


        # train for one epoch
        train(train_loader, model, loss_dict, optimizer, epoch, cfg)
        # validata for one epoch
        test(val_loader, model, cfg)




if __name__ == '__main__':
    main()
