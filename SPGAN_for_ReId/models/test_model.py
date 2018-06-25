#-*- coding:utf-8 -*-
from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def __init__(self, opt):
        assert(not opt.isTrain)
        super(TestModel, self).__init__(opt)
        self.AtoB = opt.which_direction == 'AtoB'
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B'] if self.AtoB else ['real_B', 'fake_A']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G_A'] if self.AtoB else ['G_B']
        if self.AtoB:
            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout,
                                      opt.init_type,
                                      self.gpu_ids)
        else:
            self.netG_B = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout,
                                      opt.init_type,
                                      self.gpu_ids)

    def set_input(self, input):
        # we need to use single_dataset mode
        if self.AtoB:
            self.real_A = input['A'].to(self.device)  
            self.path_A = input['A_paths']
        else:
            self.real_B =  input['B'].to(self.device)
            self.path_B = input['B_paths']

    def get_img_path(self):
        return self.path_A if self.AtoB else self.path_B 

    def forward(self):
        if self.AtoB:
            self.fake_B = self.netG_A(self.real_A)
        else:
            self.fake_A = self.netG_B(self.real_B)
