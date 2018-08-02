#-*- coding:utf-8 -*-
#===================================
# main program
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')


from PIL import Image
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import config.config as config
import reid_utils.common_utils as common_utils 
import reid_utils.model_utils as model_utils
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
    common_utils.set_devices(cfg.sys_device_ids)

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


    # create train data set
    train_loader, train_dataset = create_data_loader(cfg, cfg.trainset_part)
    # create test data set
    val_loader,_ = create_data_loader(cfg, 'val')

    # create models
    model = Model(local_conv_out_channels=128)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs for tain!")
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()


    # define loss
    loss_dict =dict(g_tri_loss = TripletLoss(margin=cfg.global_margin),
        l_tri_loss = TripletLoss(margin=cfg.local_margin))


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
        elif cfg.lr_decay_type == 'staircase_at':
            model_utils.adjust_lr_staircase_at(
                optimizer,
                cfg.base_lr,
                epoch + 1,
                cfg.staircase_decay_at_epochs,
                cfg.staircase_decay_multiply_factor)
        elif cfg.lr_decay_type == 'staircase_every':
            model_utils.adjust_lr_staircase_every(
                optimizer,
                cfg.base_lr,
                epoch + 1,
                cfg.staircase_decay_every_epochs,
                cfg.staircase_decay_multiply_factor)

        # for the purpose gradually increase the random patch 
        train_loader, _ = create_data_loader(cfg, cfg.trainset_part, epoch, cfg.total_epochs)
        # train for one epoch
        train(train_loader, model, loss_dict, optimizer, epoch, cfg)
        if (epoch+1) % cfg.val_at_epoch == 0:
            # validata for one epoch
            test(val_loader, model, cfg)



def create_data_loader(cfg, data_type, epoch=1, total_epoch=1e5):
    '''
    create the loader for train/val/test
    args:
        cfg:the object of Config
        data_type:'train','val','test' to decide the data type
    returns:
        the data loader of train/val/test data
    '''

    def img_cut_out(img):
        '''
        add radom noise to random area.
         Args:
                img (PIL Image): Image to be ''cut-out''.
        '''
        h,w = img.height, img.width
        h1 = int(h/5)
        w1 = int(w/5)
        
        stage = 1

        # get one random index
        random_h1 = random.sample(range(h),1)[0]
        random_w1 = random.sample(range(w),1)[0]

        # get another random index
        begin_h = max(random_h1 -  stage*h1, 0)
        end_h = min(random_h1 + stage*h1, h)
        begin_w = max(random_w1 -  stage*w1, 0)
        end_w = min(random_w1 + stage*w1, w)
        random_h2 = random.sample(range(begin_h, end_h),1)[0]
        random_w2 = random.sample(range(begin_w, end_w),1)[0]

        random_h = [random_h1, random_h2]
        random_h.sort()
        random_w = [random_w1, random_w2]
        random_w.sort()

        img_array = np.asarray(img)
        img_array.flags.writeable = True
        img_array[random_h[0]:random_h[1], random_w[0]:random_w[1], :] \
            = np.random.randint(0,255,size=(random_h[1]-random_h[0], random_w[1]-random_w[0], 3))
        img = Image.fromarray(np.uint8(img_array))
        return img


    if data_type == 'train' or data_type == 'trainval' :
        data_shuffle = True
        batch_size = cfg.ids_per_batch
        transform = transforms.Compose(
                            [
                            transforms.Resize(cfg.im_resize_size),
                            transforms.RandomCrop(cfg.im_crop_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.Lambda(img_cut_out),
                            transforms.RandomRotation(cfg.random_rotation_degree),
                            transforms.ToTensor(),
                            # the object of normalize should be tensor,
                            # so totensor() should called before normalize()  
                            transforms.Normalize(mean=cfg.im_mean, std=cfg.im_std)]
                            ) 
    else:
        data_shuffle = False
        batch_size=cfg.test_batch_size
        transform = transforms.Compose(
                    [
                    transforms.Resize(cfg.im_resize_size),
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
