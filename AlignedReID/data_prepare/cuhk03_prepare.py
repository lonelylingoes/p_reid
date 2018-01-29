#-*- coding:utf-8 -*-
#====================
#cuhk03数据集预处理程序
#====================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append('../')
import os
import shutil
import random
import cv2 as cv
import numpy as np

from utils import read_conf


IMAGE_WITH = read_conf.get_conf_int_param('cnn_param', 'input_with')
IMAGE_HEIGHT = read_conf.get_conf_int_param('cnn_param', 'input_height')


def create_group_data(data_dir):
    '''
    将所有图片进行组织成分组形式
    对数据进行分组处理，按以下树状结构组织：
     视角id
        --行人id
          --摄像机id
             --图片id
    '''
    #data_group={'1':{'001':{'1':[]}}}
    data_group = {}
    
    #获取目录下的所有文件名
    files = [files for files in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, files))]
    
    for file in files:
        #获取视角id,行人id，摄像机id,图片id
        viewId, personId, cameraId, pictureId = file.split('_')
        pictureId,_ = pictureId.split('.')

        #挨个判断每层key是否存在
        if viewId not in data_group.keys():
            d = {personId:{cameraId:[pictureId]}}
            data_group[viewId] = d
        elif personId not in data_group[viewId].keys():
            d = {cameraId:[pictureId]}
            data_group[viewId][personId] = d
        elif cameraId not in data_group[viewId][personId].keys():
            d = [pictureId]
            data_group[viewId][personId][cameraId] = d
        else:
            data_group[viewId][personId][cameraId].append(pictureId)#没有keys时，不支持直接这样添加的写法,只能挨个判断
            
    print("there are %d files, group data done!" % len(files))
    
    return data_group



def devide_group_data_set(group_data):
    '''
    根据CUHK03数据集的特点，将数据集划分为train_set|validation_set|test_set
    其中，val_set和test_set分别为100,100,person,按比例分别从各个view中随机取出
    将各个camera拍到的行人按照两两匹配的方式进行配对
    input:
        group_data:最原始的分组数据
    output:
        train_set:训练数据组集,结构与group_data类似
        val_set:验证数据组集,结构与group_data类似
        test_set:测试数据组集,结构与group_data类似
    '''
    total_person_num = 0
    val_set_num = 100
    test_set_num = 100
    test_set={}
    val_set = {}
    
    view_personNum_list=[]#视角和其人数组成的元祖列表
    for view in group_data.keys():
        print("in view%s there are %d persons" % (view, len(group_data[view].keys())))
        view_personNum_list.append((view, len(group_data[view].keys())))
        total_person_num += len(group_data[view].keys())

    #按比例取样，并去除采样过的
    for view_personNum in view_personNum_list:
        viewId, person_num = view_personNum#view id 和该id下的人数
        psersionID_list = group_data[viewId].keys()#一个view中所有person的列表
        
        sampe_num =  int(person_num/total_person_num * val_set_num + 0.5)
        val_list = random.sample(psersionID_list,sampe_num)#按比例随机采样验证数据
        #--取字典子集--将采样出来的数据结按照相同的结构存放到验证集中
        val_person = {key:group_data[viewId][key] for key in group_data[viewId].keys() if key in val_list}
        val_set[viewId] = val_person
        group_data[viewId] = {key:group_data[viewId][key] for key in group_data[viewId].keys() if key not in val_list}#将原始数据集中val数据去除
        
        sampe_num = int(person_num/total_person_num * test_set_num + 0.5)
        test_list = random.sample(psersionID_list, sampe_num)#采样测试数据
        #--取字典子集--将采样出来的数据结按照相同的结构存放到验证集中
        test_person = {key:group_data[viewId][key] for key in group_data[viewId].keys() if key in test_list}
        test_set[viewId] = test_person
        group_data[viewId] = {key:group_data[viewId][key] for key in group_data[viewId].keys() if key not in test_list} #将原始数据集中val数据去除
        
    train_set = group_data
    
    return train_set, val_set, test_set



def joint_id_to_name(viewId, personId, cameraId, pictureId):
    '''
    将路径和各种id拼接成全路径名
    '''
    file_name =  viewId + '_' + personId + '_' + cameraId + '_' + pictureId + '.png'
    return  file_name



def prepare_data(data_dir):
    '''
    数据准备，将数据集创建相应的文件夹并移动
    '''
    group_data = create_group_data(data_dir)
    train_set, val_set, test_set = devide_group_data_set(group_data)

    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir) 
    move_file(train_set, data_dir, train_dir)

    val_dir = os.path.join(data_dir, "val")
    if not os.path.exists(val_dir):
        os.makedirs(val_dir) 
    move_file(val_set, data_dir, val_dir)

    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir) 
    move_file(test_set, data_dir, test_dir)



def move_file(data_set, src_dir, dst_dir):
    '''
    移动文件到指定文件夹
    '''
    for viewId in data_set.keys():
        for personId in data_set[viewId].keys():
            for cameraId in data_set[viewId][personId].keys():
                for pictureId in data_set[viewId][personId][cameraId]:
                    src_file = os.path.join(src_dir, joint_id_to_name(viewId, personId, cameraId, pictureId))
                    dest_file = os.path.join(dst_dir, joint_id_to_name(viewId, personId, cameraId, pictureId))
                    shutil.move(src_file, dest_file)  



def reover(data_dir):
    '''
    恢复数据到原始状态
    '''

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    shutil.copytree(train_dir,data_dir,False)
    shutil.retree(train_dir)
    shutil.copytree(val_dir,data_dir,False)
    shutil.retree(val_dir)
    shutil.copytree(test_dir,data_dir,False)
    shutil.retree(test_dir)
    
    
    
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
    print('there are %d pictures in chuhk03' % len(filenames))
    total_num = 0#该数据集中行人总数
    for file in filenames:
        #获取视角id,行人id，摄像机id,图片id
        name = file.split('_')
        personId = name[1]
        if personId not in person_dict.keys():
            person_dict[personId] = total_num
            total_num += 1
        
    for i in range(total_num):
        picture_list.append([])
    
    for file in filenames:
        name = file.split('_')
        personId = name[1]
        
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
        personId = name[1]
        if personId not in person_dict.keys():
            person_dict[personId] = total_num
            total_num += 1
        
    for i in range(total_num):
        picture_list.append([])
    
    for file in filenames:
        name = file.split('_')
        personId = name[1]
        
        if len(picture_list[person_dict[personId]]) == 0:
            picture_list[person_dict[personId]] = [(person_dict[personId], os.path.join(data_dir, file))]
        else:
            picture_list[person_dict[personId]].append((person_dict[personId], os.path.join(data_dir, file)))
        
    del filenames
    del person_dict
        
    dest_dir = os.path.join(data_dir, 'bin_file')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir) 
    turn_file_to_bin(picture_list, identity_num, dest_dir, 'cuhk03_train')
    
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
        
        
