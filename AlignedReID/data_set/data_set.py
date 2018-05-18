#-*- coding:utf-8 -*-
#===================================
# data set program
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from  os.path import expanduser as ospeu
import torchvision.transforms as transforms

import random
import numpy as np
from reid_utils.common_utils import load_pickle
from reid_utils.dataset_utils import parse_full_path_market1501_im_name
from reid_utils.dataset_utils import parse_full_path_duke_im_name
from reid_utils.dataset_utils import parse_full_path_new_im_name
from collections import defaultdict
from PIL import Image


class ReIdDataSet(Dataset):
    '''
    ReIdDataSet class 
    '''
    def __init__(self, 
        data_type, 
        cfg,
        transform = None):
        '''
        args:
            data_type: indicate the kind of data will be created
            cfg:    config object
            transform: transform object collection
        '''
        assert data_type in ['trainval', 'train', 'val', 'test']

        if data_type == 'trainval':
            partition = load_pickle(ospeu(cfg.train_dataset_partitions))
            if  cfg.train_dataset == 'market1501':
                self.parse_full_path_im_name = parse_full_path_market1501_im_name
            elif cfg.train_dataset == 'duke':
                self.parse_full_path_im_name = parse_full_path_duke_im_name
            elif cfg.train_dataset == 'cuhk03':
                self.parse_full_path_im_name =  parse_full_path_new_im_name
            elif cfg.train_dataset == 'combine':
                pass

            self.ims_per_id = cfg.ims_per_id
            self.ims_names = partition['trainval_im_names']
            self.ids2labels = partition['trainval_ids2labels']

            im_ids = [self.parse_full_path_im_name(name, 'id') for name in self.ims_names]
            self.ids_to_im_indexs = defaultdict(list)
            for index, id in enumerate(im_ids):
                self.ids_to_im_indexs[id].append(index)
            # id list
            self.ids = self.ids_to_im_indexs.keys()
        elif data_type == 'train':
            partition = load_pickle(ospeu(cfg.train_dataset_partitions))
            if  cfg.train_dataset == 'market1501':
                self.parse_full_path_im_name = parse_full_path_market1501_im_name
            elif cfg.train_dataset == 'duke':
                self.parse_full_path_im_name = parse_full_path_duke_im_name
            elif cfg.train_dataset == 'cuhk03':
                self.parse_full_path_im_name =  parse_full_path_new_im_name
            elif cfg.train_dataset == 'combine':
                pass

            self.ims_per_id = cfg.ims_per_id
            self.ims_names = partition['train_im_names']
            self.ids2labels = partition['train_ids2labels']

            im_ids = [self.parse_full_path_im_name(name, 'id') for name in self.ims_names]
            self.ids_to_im_indexs = defaultdict(list)
            for index, id in enumerate(im_ids):
                self.ids_to_im_indexs[id].append(index)
            # id list
            self.ids = self.ids_to_im_indexs.keys()
        elif data_type == 'val':
            partition = load_pickle(ospeu(cfg.test_dataset_partitions))
            if  cfg.train_dataset == 'market1501':
                self.parse_full_path_im_name = parse_full_path_market1501_im_name
            elif cfg.train_dataset == 'duke':
                self.parse_full_path_im_name = parse_full_path_duke_im_name
            elif cfg.train_dataset == 'cuhk03':
                self.parse_full_path_im_name =  parse_full_path_new_im_name
            elif cfg.train_dataset == 'combine':
                pass

            self.ims_names = partition['val_im_names']
            self.marks = partition['val_marks']
        elif data_type == 'test':
            partition = load_pickle(ospeu(cfg.test_dataset_partitions))
            if  cfg.train_dataset == 'market1501':
                self.parse_full_path_im_name = parse_full_path_market1501_im_name
            elif cfg.train_dataset == 'duke':
                self.parse_full_path_im_name = parse_full_path_duke_im_name
            elif cfg.train_dataset == 'cuhk03':
                self.parse_full_path_im_name =  parse_full_path_new_im_name
            elif cfg.train_dataset == 'combine':
                pass

            self.ims_names = partition['test_im_names']
            self.marks = partition['test_marks']
        else:
            pass

        self.transform = transform
        self.data_type = data_type
        
        
    def __len__(self):
        '''
        the length of the data set is the ids' number.
        '''
        if self.data_type in ['trainval', 'train']:
            return len(self.ids)
        else:
            return len(self.ims_names)


    def __getitem__(self, index_of_item):
        '''
        for training, one sample means several images (and labels etc) of one id.
        for testing, one sample means one image
        args:
            index_of_item: for training, it is the index of ids; for testing, it is the index of images
        '''
        if self.data_type in ['trainval', 'train']:
            # get the specified id's index list
            indexs = self.ids_to_im_indexs[self.ids[index_of_item]]
            if len(indexs) < self.ims_per_id:
                indexs = np.random.choice(indexs, self.ims_per_id, replace=True)
            else:
                indexs = np.random.choice(indexs, self.ims_per_id, replace=False)
            ims_names = [self.ims_names[index] for index in indexs]
            ims = [Image.open(name) for name in ims_names]# fro transform the imput type must be PIL type
            # image aurgment
            if self.transform is not None:
                ims = [self.transform(im).numpy() for im in ims]
            # labels are same
            labels = [self.ids2labels[self.ids[index_of_item]] for _ in range(self.ims_per_id)]
            return np.array(ims), np.array(labels)
        else:
            im_name = self.ims_names[index_of_item]
            im = Image.open(im_name)
            # image aurgment
            if self.transform is not None:
                im = self.transform(im)
            id = self.parse_full_path_im_name(self.ims_names[index_of_item], 'id')
            cam = self.parse_full_path_im_name(self.ims_names[index_of_item], 'cam')
            # denoting whether the im is from query, gallery, or multi query set
            mark = self.marks[index_of_item]
            return im, id, cam, mark