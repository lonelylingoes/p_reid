#-*- coding:utf-8 -*-
#====================
#cuhk-sysu数据集预处理程序
#====================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.io as sio
import os
from PIL import Image
import random
import cv2 as cv
import numpy as np
from utils import read_conf


IMAGE_WITH = read_conf.get_conf_int_param('cnn_param', 'input_with')
IMAGE_HEIGHT = read_conf.get_conf_int_param('cnn_param', 'input_height')
DATA_FORMAT = read_conf.get_conf_str_param('cnn_param', 'data_format')



def prepare_data(data_dir):
    '''
    数据准备，将数据集创建相应的文件夹并移动
    args:
        data_dir:数据集根级文件的存放路径。
    '''
    get_train_images(data_dir)
    get_test_images(data_dir)




def get_test_images(data_dir):
    '''
    获取测试图片
    '''
    # 获取测试图片集
    
    src_dir = os.path.join(data_dir, "Image/SSM")

    query_dir = os.path.join(data_dir, "Image/query")
    if not os.path.exists(query_dir):
        os.makedirs(query_dir) 

    gallery_dir = os.path.join(data_dir, "Image/gallery")
    if not os.path.exists(gallery_dir):
        os.makedirs(gallery_dir) 

    images_data = sio.loadmat(os.path.join(data_dir, 'annotation/Images.mat'))
    images_data = images_data['Img']
    #images_data[:,i]为列表数据，里面就一个元祖数据
    #images_data[:,i][0] 为元祖数据
    #images_data[:,i][0][0]为图片名
    #images_data[:,i][0][1]为人个数
    #images_data[:,i][0][2]为个人坐标数组
    images_dict = {}
    for i in range(images_data.shape[1]):
        per_image = images_data[:,i]
        image_name = per_image[0][0][0]
        identity_num = per_image[0][1][0][0]
        index = random.randint(0, identity_num-1)#随机取每个图片中包含的一个人,用在组成gallery的时候
        images_dict[image_name] = per_image[0][2][0][index][0][0]#组成坐标[x, y, w, h]
    

    test_data = sio.loadmat(os.path.join(data_dir, 'annotation/test/train_test/TestG50.mat'))
    test_data = test_data['TestG50']    
    #test_data[:,i]为列表数据，里面就一个元祖数据
    #test_data[:,i][0] 为元祖数据
    #test_data[:,i][0][0]为query信息
    #test_data[:,i][0][0][0][0]为query信息元祖
    #test_data[:,i][0][0][0][0][0]为query图片名
    #test_data[:,i][0][0][0][0][1]为query中人的坐标
    #test_data[:,i][0][0][0][0][3]为query中人的id
    #test_data[:,i][0][1]为gallery信息二维数组
    #test_data[:,i][0][1][0][k][0]为gallery信息中第k个的图片名
    #test_data[:,i][0][1][0][k][1]为gallery信息中第k个的坐标，为空则表示该id没出现在该图片中
    for i in range(test_data.shape[1]):
        per_test = test_data[:,i]
        person_id = per_test[0][0][0][0][3][0]
        x = int(per_test[0][0][0][0][1][0][0])
        y = int(per_test[0][0][0][0][1][0][1])
        w = int(per_test[0][0][0][0][1][0][2])
        h = int(per_test[0][0][0][0][1][0][3])
        new_name = person_id + '.jpg'#获取query新文件名
        im = Image.open(os.path.join(src_dir, per_test[0][0][0][0][0][0]))
        region = im.crop((x, y, x+w, y+h))
        region.save(os.path.join(query_dir, new_name))


        # 生成gallery图片集
        gallery_num = len(per_test[0][1][0])
        for j in range(gallery_num):
            src_file = per_test[0][1][0][j][0][0]#图片文件名
            if len(per_test[0][1][0][j][1][0]) != 0:
                new_name = person_id + '_%d_p.jpg'%(j+1)#获取query新文件名
                x = int(per_test[0][1][0][j][1][0][0])
                y = int(per_test[0][1][0][j][1][0][1])
                w = int(per_test[0][1][0][j][1][0][2])
                h = int(per_test[0][1][0][j][1][0][3])
            else:#没有待查询的人，随机找一个人生成图片
                new_name = person_id + '_%d_n.jpg'%(j+1)#获取query新文件名
                box = images_dict[src_file]
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3])
            
            im = Image.open(os.path.join(src_dir, src_file))
            region = im.crop((x, y, x+w, y+h))
            region.save(os.path.join(gallery_dir, new_name))



def get_train_images(data_dir):
    '''
    获取训练图片
    '''
    src_dir = os.path.join(data_dir, "Image/SSM")

    train_dir = os.path.join(data_dir, "Image/train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir) 

    # 获取训练图片集
    train_data = sio.loadmat(os.path.join(data_dir, 'annotation/test/train_test/Train.mat'))
    train_data = train_data['Train']
    #train_data[i,:]为列表数据，就一个元素为二维数组
    #train_data[i,:][0][0][0] 为元祖数据
    #train_data[i,:][0][0][0][0]为人的id列表
    #train_data[i,:][0][0][0][1]为人出现在图片的个数数组
    #train_data[i,:][0][0][0][2]为个人出现在图片中的信息数组
    #train_data[i,:][0][0][0][2][0]为个人出现在图片中的信息数组
    #train_data[i,:][0][0][0][2][0][j][0]为个人出现在图片j中的信息的图片名
    #train_data[i,:][0][0][0][2][0][j][1]为个人出现在图片j中的信息的坐标
    # 对每个identity从图片中截取出来存放到
    for i in range(train_data.shape[0]):
        per_train = train_data[i,:]
        person_id = per_train[0][0][0][0][0]
        appear_num = per_train[0][0][0][1][0][0]
        for j in range(appear_num):
            new_name = person_id + '_%d.jpg'%(j+1)#获取新文件名
            im = Image.open(os.path.join(src_dir, per_train[0][0][0][2][0][j][0][0]))
            x = int(per_train[0][0][0][2][0][j][1][0][0])
            y = int(per_train[0][0][0][2][0][j][1][0][1])
            w = int(per_train[0][0][0][2][0][j][1][0][2])
            h = int(per_train[0][0][0][2][0][j][1][0][3])
            region = im.crop((x, y, x+w, y+h))
            region.save(os.path.join(train_dir, new_name))

  
    
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
    filenames = [files for files in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, files))]
    print('there are %d pictures in cuhk_sysu' % len(filenames))
    total_num = 0#该数据集中行人总数
    for file in filenames:
        #获取视角id,行人id，摄像机id,图片id
        name = file.split('_')
        personId = name[0]
        if personId not in person_dict.keys():
            person_dict[personId] = total_num
            total_num += 1
        
    for i in range(total_num):
        picture_list.append([])
    
    for file in filenames:
        name = file.split('_')
        personId = name[0]
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
    filenames = [files for files in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, files))]
    print('there are %d pictures in chuhk03' % len(filenames))
    total_num = 0#该数据集中行人总数
    for file in filenames:
        #获取视角id,行人id，摄像机id,图片id
        name = file.split('_')
        personId = name[0]
        if personId not in person_dict.keys():
            person_dict[personId] = total_num
            total_num += 1
        
    for i in range(total_num):
        picture_list.append([])
    
    for file in filenames:
        name = file.split('_')
        personId = name[0]
        if len(picture_list[person_dict[personId]]) == 0:
            picture_list[person_dict[personId]] = [(person_dict[personId], os.path.join(data_dir, file))]
        else:
            picture_list[person_dict[personId]].append((person_dict[personId], os.path.join(data_dir, file)))
        
    del filenames
    del person_dict
        
    dest_dir = os.path.join(data_dir, 'bin_file')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    turn_file_to_bin(picture_list, identity_num, dest_dir, 'cuhk_sysu_train')
    
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
        

