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
from model.model import FirstStageModel, Model
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
    
    # test on test set
    if cfg.only_test:
        test_loader, _ = create_data_loader(cfg, 'test')
        # create models
        model = Model(local_conv_out_channels=128, pretrained=False)
        # load model param
        model = model_utils.load_test_model(model, cfg)
        # after load model, parallel the model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs for test!")
            model = nn.DataParallel(model)
        if torch.cuda.is_available():
            model.cuda()
        # just for test
        test(test_loader, model, cfg)
        return

    ############# first stage #############

    # create train data set
    train_loader, train_dataset = create_data_loader(cfg, 'train')
    # create test data set
    val_loader,_ = create_data_loader(cfg, 'val')
    # create model
    model = FirstStageModel(num_classes=len(train_dataset.ids2labels))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs for tain!")
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()

    # define loss
    id_criterion = nn.CrossEntropyLoss().cuda() 



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
    #cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

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


    ############# second stage #############
    g_tri_loss = TripletLoss(margin=cfg.global_margin),

def create_data_loader(cfg, data_type):
    '''
    create the loader for train/val/test, 
    create data loader for the first stage.
    args:
        cfg:the object of Config
        data_type:'train','val','test' to decide the data type
    returns:
        the data loader of train/val/test data
    '''
    if data_type == 'train':
        data_shuffle = True
        batch_size = cfg.first_stage_batch_size
        transform = transforms.Compose(
                            [transforms.Resize(cfg.first_resize_size),
                            transforms.RandomCrop(cfg.first_crop_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            # the object of normalize should be tensor,
                            # so totensor() should called before normalize()  
                            transforms.Normalize(mean=cfg.im_mean, std=cfg.im_std)]
                            ) 
    else:
        data_shuffle = False
        batch_size=cfg.test_batch_size
        transform = transforms.Compose(
                    [transforms.Resize(cfg.first_crop_size),
                    transforms.ToTensor(),
                    # the object of normalize should be tensor,
                    # so totensor() should called before normalize() 
                    transforms.Normalize(mean=cfg.im_mean, std=cfg.im_std)]
                    )

    dataset = ReIdDataSet(data_type,
                        cfg,
                        transform)
    data_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle = data_shuffle,
                        num_workers=cfg.workers, pin_memory=True)


    return data_loader, dataset





if __name__ == '__main__':
    main()
