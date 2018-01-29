#-*- coding:utf-8 -*-
#====================
#数据增强程序
#====================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

import tensorflow as tf
import random
from scipy.special import comb
import os
from utils import read_conf
import scipy



 
def random_crop(input):
    '''
    随机水平裁剪[height, width]的图片
    args:
        input:
    returns:

    '''
    crop_rate = read_conf.get_conf_float_param('augment_param', 'crop_rate')
    h = int(input.shape[0].value * crop_rate)
    w = int(input.shape[1].value * crop_rate)
        
    return tf.random_crop(input, [h, w, 3])
        


def random_flip_left_right(input):
    '''
    随机左右翻转图像
    args:
        input
    returns:

    '''
    return tf.image.random_flip_left_right(input)



def random_brightness(input):
    '''
    随机增加亮度
     args:
         input
     returns:
     '''
    return tf.image.random_brightness(input, max_delta=63)    



def random_random_contrast(input):
    '''
     随机增加对比度
     args:
         input
     returns:
    '''
    return  tf.image.random_contrast(input, lower=0.2, upper=1.8)
    


def single_augment(input):
    '''
     对单个图片进行增广
     args:
         input
     returns:
        增广后的输出数据列表
    '''
    augment_num = read_conf.get_conf_int_param('augment_param', 'augment_num')
    
    #获取数据增广的函数列表
    augment_function_list = []
    random_crop_flag = read_conf.get_conf_bool_param('augment_param', 'random_crop')
    if random_crop_flag:
        augment_function_list.append(random_crop)

    random_flip_left_right_flag = read_conf.get_conf_bool_param('augment_param', 'random_flip_left_right')
    if random_flip_left_right_flag:
        augment_function_list.append(random_flip_left_right)

    random_brightness_flag = read_conf.get_conf_bool_param('augment_param', 'random_brightness')
    if random_brightness_flag:
        augment_function_list.append(random_brightness)

    random_random_contrast_flag = read_conf.get_conf_bool_param('augment_param', 'random_random_contrast')
    if random_random_contrast_flag:
        augment_function_list.append(random_random_contrast)

    #提醒可能的异常
    augment_type_num = len(augment_function_list)
    posibale_augment_num = 0#可能的增广方式数量
    for i in range(1, augment_type_num+1):
        posibale_augment_num += comb(augment_type_num, i)
    if posibale_augment_num < augment_num:
        raise ValueError("===error: the posibale_augment_num is less than augment_num, there will be some repeat images.")


    augmented_input = []
    augment_function_used = []#使用过的增强手段
    for i in range(augment_num):
        #确保随机增广的方式没有重复，不会生成重复图片
        while True:
            augment_num =  random.randint(1, augment_type_num)#生成一个随机数，用来确定采用增强手段的种数
            augment_function_index = random.sample(range(augment_type_num), augment_num)#从所有增强手段中随机取出n个
            if augment_function_index not in augment_function_used:#增广方式没有用过,则采用这次生成的增广方式
                break
        augment_function_used.append(augment_function_index)

        aug_input = input
        for j in augment_function_index:
            aug_input = augment_function_list[j](aug_input)
        augmented_input.append(aug_input)

    return augmented_input




def augment(data_dir):
    '''
    对目录内的每个图片进行数据增广处理
    '''
    
    files = []
    for file_info in os.walk(data_dir):
        for file in file_info[2]:
            files.append(os.path.join(file_info[0], file))
    
    for file in files:
        with tf.Graph().as_default():#指定计算图谱的范围
            input = tf.read_file(file)
            input = tf.image.decode_image(input, channels=3)
            with tf.Session() as sess:
                input = sess.run(input)#这是关键，没有放到session里run数据这些tensor都是没数据的
                input = tf.cast(input, tf.float32)
                augmented_input = single_augment(input)
                #生成增广文件
                for i in range(len(augmented_input)):
                    full_path_name, extend_name = os.path.splitext(file)
                    output = tf.cast(augmented_input[i], tf.uint8)
                    new_name = (full_path_name + '_aug_' +'%d' + extend_name) % (i+1)
                    scipy.misc.imsave(new_name, output.eval())
                
    print(data_dir, "augment done!")
