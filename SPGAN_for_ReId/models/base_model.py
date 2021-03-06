#-*- coding:utf-8 -*-

import os
import torch
from collections import OrderedDict
from . import networks


class BaseModel(object):
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def setup(self, opt):
        '''load the model's checkpoint; print networks; create shedulars, get epoch_count.'''
        if not self.isTrain or opt.continue_train:
            opt.epoch_count = self.load_networks(opt.which_epoch)
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        self.print_networks(opt.verbose)

    def eval(self):
        ''' make models eval mode during test time'''
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        ''' used in test time, wrapping `forward` in no_grad() so we don't save
        intermediate steps for backprop
        '''
        self.eval()
        with torch.no_grad():
            self.forward()

    def optimize_parameters(self):
        pass

    def update_learning_rate(self):
        ''' update learning rate (called once every epoch)'''
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        '''return visualization images.
        train.py will display these images, and save the images to a html
        '''
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        '''return traning losses/errors. 
        train.py will print out these errors as debugging information
        '''
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, which_epoch, epoch_value):
        '''save models to the disk'''
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # add .module, otherwise the name will add 'module'
                    ckpt = dict(state_dict=net.module.state_dict(),
                            epoch=epoch_value)
                else:
                    ckpt = dict(state_dict=net.cpu().state_dict(),
                            epoch=epoch_value)
                torch.save(ckpt, save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, which_epoch):
        '''load models from the disk'''
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                checkpoint = torch.load(load_path, map_location=str(self.device))
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(checkpoint['state_dict'].keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(checkpoint['state_dict'], net, key.split('.'))
                net.load_state_dict(checkpoint['state_dict'])
        return checkpoint['epoch']


    def print_networks(self, verbose):
        '''print network information'''
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')


    def set_requires_grad(self, nets, requires_grad=False):
        '''set requies_grad=Fasle to avoid computation'''
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
