#-*- coding:utf-8 -*-
#====================
#数据集预处理程序
#====================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os
import argparse
import sys
import cuhk03_prepare
import market1501_prepare
import mars_prepare
import cuhk_sysu_prepare
import data_augment 

from utils import read_conf

FLAGS = None


def main(_):
        
    data_dir = {'CUHK03' : '../../data_set/CUHK03/cuhk03_labeled/',
        'Market1501' : '../../data_set/Market1501/',
        'MARS' : '../../data_set/MARS/',
        'CUHK-SYSU' : '../../data_set/CUHK-SYSU/dataset/'}
    dest_file = '../../data_set/train.txt'
    
    # 将数据进行划分，已经划分的数据集不要操作
# =============================================================================
#     cuhk03_prepare.prepare_data(data_dir['CUHK03'])
#     cuhk_sysu_prepare.prepare_data(data_dir['CUHK-SYSU'])
# =============================================================================
    
    # 训练数据集路径
    cuhk03_train_path = os.path.join(data_dir['CUHK03'], 'train')
    market501_train_path = os.path.join(data_dir['Market1501'], 'bounding_box_train')
    mars_train_path = os.path.join(data_dir['MARS'], 'bbox_train')
    cuhk_sysu_train_path = os.path.join(os.path.join(data_dir['CUHK-SYSU'], 'Image'), 'train')
    
    # 数据增广
# =============================================================================
#     data_augment.augment(cuhk03_train_path)
#     data_augment.augment(market501_train_path)
#     data_augment.augment(mars_train_path)
#     data_augment.augment(cuhk_sysu_train_path)
# =============================================================================
    
    #写文件名和label到文件
    identity_num = 0
    identity_num = cuhk03_prepare.write_picture_label_to_text(cuhk03_train_path, dest_file,identity_num)  
    print('after cuhk03 identity_num=', identity_num)
    identity_num = market1501_prepare.write_picture_label_to_text(market501_train_path, dest_file, identity_num)
    print('after market1501 identity_num=', identity_num)
    identity_num = cuhk_sysu_prepare.write_picture_label_to_text(cuhk_sysu_train_path, dest_file, identity_num)
    print('after sysu identity_num=', identity_num)
    #identity_num = mars_prepare.write_picture_label_to_text(mars_train_path, identity_num) 
    #print('after mars identity_num=:', identity_num)
    
    # 数据组成batch提取成tf的读取形式
    identity_num = 0
    identity_num = cuhk03_prepare.create_train_file(cuhk03_train_path, identity_num)  
    print('after cuhk03 identity_num=', identity_num)
    identity_num = market1501_prepare.create_train_file(market501_train_path, identity_num)
    print('after market1501 identity_num=', identity_num)
    identity_num = cuhk_sysu_prepare.create_train_file(cuhk_sysu_train_path, identity_num)
    print('after cuhk_sysu identity_num=', identity_num)
    #identity_num = mars_prepare.create_train_file(mars_train_path, identity_num) 
    #print('after mars identity_num=:', identity_num)
    #将人的总数写入配置文件
    
    read_conf.set_conf_param('hyper_param', 'identities', identity_num)
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)