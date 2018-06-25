#-*- coding:utf-8 -*-
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    def __init__(self, opt):
        super(SingleDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        AtoB = opt.which_direction == 'AtoB'
        sub_dir = opt.sub_dirA if AtoB else opt.sub_dirB
        self.dir = os.path.join(opt.dataroot, sub_dir)

        self.paths = make_dataset(self.dir)
        self.paths = sorted(self.paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        transImg = self.transform(img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = transImg[0, ...] * 0.299 + transImg[1, ...] * 0.587 + transImg[2, ...] * 0.114
            transImg = tmp.unsqueeze(0)

        return {'A': transImg, 'A_paths': path} if self.opt.which_direction == 'AtoB' \
                else  {'B': transImg, 'B_paths': path} 

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'SingleImageDataset'
