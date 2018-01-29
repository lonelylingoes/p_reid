#-*- coding:utf-8 -*-
#====================
#mars数据集预处理程序
#====================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('../')
import os
import random
import cv2 as cv
import numpy as np

from utils import read_conf


IMAGE_WITH = read_conf.get_conf_int_param('cnn_param', 'input_with')
IMAGE_HEIGHT = read_conf.get_conf_int_param('cnn_param', 'input_height')



def write_picture_label_to_text(data_dir, dest_file, identity_num):
    '''
    将图片转成tf读取的二进制格式数据
    args:
        data_dir:训练数据路径
        dest_file:最终存放图片和label全路径文件名
        identity_num:外部传入的累计行人的个数，用来确定分类数
    returns:
        identity 累计总个数
    '''

    person_dict = {}#行人ID和行人序号构成的自定{'id':序号}，该序号刚好是picture_list中元素的索引
    picture_list=[]#图片名列表[[(序号，p1),(序号，p2), ...], ...],将索引存入数据方便后续随机选取数据
    
    #获取目录下的所有文件名
    #递归获取目录下的所有文件名
    filenames = []
    for file_info in os.walk(data_dir):
        for file in file_info[2]:
            filenames.append(os.path.join(file_info[0], file))
    print('there are %d pictures in mars' % len(filenames))
    total_num = 0#该数据集中行人总数
    for file in filenames:
        #行人id
        name = os.path.basename(file)#从全路径中获取文件名
        personId = name[0:4]
        if personId not in person_dict.keys():
            person_dict[personId] = total_num
            total_num += 1
        
    for i in range(total_num):
        picture_list.append([])
    
    for file in filenames:
        #行人id
        name = os.path.basename(file)#从全路径中获取文件名
        personId = name[0:4]
        if len(picture_list[person_dict[personId]]) == 0:
            picture_list[person_dict[personId]] = [(person_dict[personId], os.path.join(data_dir, file))]
        else:
            picture_list[person_dict[personId]].append((person_dict[personId], os.path.join(data_dir, file)))
        
    del filenames
    del person_dict
        
    images_per_id = read_conf.get_conf_int_param('hyper_param', 'images_per_id')
    with open(dest_file, 'a+') as f:
        # 数据转换直到picture_list为空
        while len(picture_list) != 0:
            index = random.randint(0, len(picture_list)-1)#随机选取行人
            if len(picture_list[index]) >= images_per_id:#不够采样要求了删除整个人的图片列表
                sample_pictures = random.sample(picture_list[index], images_per_id)#从人的图片中随机选择
                #写入文件
                for pictures in sample_pictures:
                    f.write(pictures[1])
                    f.write(' ')
                    f.write(str(pictures[0]))
                    f.write('\n')
                    f.flush()
                # 删除已经使用过的
                picture_list[index] = list(set(picture_list[index]).difference(set(sample_pictures)))
            else:
                picture_list.pop(index)
    
    return total_num + identity_num  



def create_train_file(data_dir, identity_num):
    '''
    将图片转成tf读取的二进制格式数据
    args:
        data_dir:训练数据路径
        identity_num:外部传入的累计行人的个数，用来确定分类数
    returns:
        identity 累计总个数
    '''
    person_dict = {}#行人ID和行人序号构成的自定{'id':序号}，该序号刚好是picture_list中元素的索引
    picture_list=[]#图片名列表[[(序号，p1),(序号，p2), ...], ...],将索引存入数据方便后续随机选取数据
    
    #获取目录下的所有文件名
    #递归获取目录下的所有文件名
    filenames = []
    for file_info in os.walk(data_dir):
        for file in file_info[2]:
            filenames.append(os.path.join(file_info[0], file))
    print('there are %d pictures in mars' % len(filenames))
    total_num = 0#该数据集中行人总数
    for file in filenames:
        #行人id
        name = os.path.basename(file)#从全路径中获取文件名
        personId = name[0:4]
        if personId not in person_dict.keys():
            person_dict[personId] = total_num
            total_num += 1
        
    for i in range(total_num):
        picture_list.append([])
    
    for file in filenames:
        #行人id
        name = os.path.basename(file)#从全路径中获取文件名
        personId = name[0:4]
        if len(picture_list[person_dict[personId]]) == 0:
            picture_list[person_dict[personId]] = [(person_dict[personId], os.path.join(data_dir, file))]
        else:
            picture_list[person_dict[personId]].append((person_dict[personId], os.path.join(data_dir, file)))
        
    del filenames
    del person_dict
        
    dest_dir = os.path.join(os.path.join(data_dir, '../'), 'bin_file')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    turn_file_to_bin(picture_list, identity_num, dest_dir, 'mars_train')
    
    return total_num + identity_num



def turn_picture_to_bin(picture_file):
    '''
    将图片转换成二进制格式写入文件
    input:
        picture_file:图片文件名
    output:
        输出待写入的np.array
    '''
    img = cv.imread(picture_file)
    img = cv.resize(img,(IMAGE_WITH, IMAGE_HEIGHT))
    img = img[:,:,[2,1,0]]#BGR ==>RGB
    l = []
    #RRRRGGGGBBBBB这样的顺序
    l.extend(img[:,:,0].reshape(IMAGE_HEIGHT*IMAGE_WITH, order='C'))
    l.extend(img[:,:,1].reshape(IMAGE_HEIGHT*IMAGE_WITH, order='C'))
    l.extend(img[:,:,2].reshape(IMAGE_HEIGHT*IMAGE_WITH, order='C'))
    
    return np.array(l, np.uint16)



def turn_file_to_bin(picture_list,identity_num, 
                         dest_path, dest_name, file_limit=400):
    '''
    数据集中的将文件转换成二进制文件
    input:
        picture_list:全路径图片数据
        identity_num:外部传入的累计行人的个数，用来确定分类数
        dest_path:二进制文件存放路径
        dest_name:转换后生成的文件名前缀，
        file_limit:转换后的文件大小限制，超过此文件大小将会自动分割,单位M
    '''
    images_per_id = read_conf.get_conf_int_param('hyper_param', 'images_per_id')
    serial_number = 1
    new_file_name = os.path.join(dest_path, dest_name)  + '_' + str(serial_number)+ ".bin"
    
    f = None
    try:
        f = open(new_file_name, 'wb')
        
        # 数据转换直到picture_list为空
        while len(picture_list) != 0:
            index = random.randint(0, len(picture_list)-1)#随机选取行人
            if len(picture_list[index]) >= images_per_id:#不够采样要求了删除整个人的图片列表
                sample_pictures = random.sample(picture_list[index], images_per_id)#从人的图片中随机选择
                #写入文件
                for pictures in sample_pictures:
                    f.write(np.uint16(pictures[0]) + np.uint16(identity_num))
                    f.write(turn_picture_to_bin(pictures[1]))
                    f.flush()
                # 删除已经使用过的
                picture_list[index] = list(set(picture_list[index]).difference(set(sample_pictures)))
            else:
                picture_list.pop(index)
        
            #一个图片对写完了判断文件大小,超过限制重新创建文件
            if os.path.getsize(new_file_name)/1024.0/1024.0 > file_limit:
                f.close()
                serial_number += 1
                new_file_name = os.path.join(dest_path, dest_name)  + '_' + str(serial_number)+ ".bin"
                f = open(new_file_name, 'wb')
    finally:
        if f:
            f.close()
        

