# -*- coding: utf-8 -*-

#===============================
#文件预处理程序：
#    将图片数据以文件名提供的规律，提取成positive pair
#    同时将图片尺寸归一到同一尺寸
#===============================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import cv2 as cv 
import os
import argparse
import sys
import random
import numpy as np
from fnmatch import fnmatch

FLAGS = None
IMAGE_WITH = 66 #变成输入网络数据的10%
IMAGE_HEIGHT = 176 #变成输入网络数据的10%
POSITIVE_NEGATIVE_RATE = 2 #负正样本比例，按照论文值


def main(_):
    
    #根据图片特点创建分组结构
    group_data = create_group_data(FLAGS.data_dir)
    
    #将数据进行组对：positive, negative
    positive_pair_set, negative_pair_set = create_data_pair(group_data)

    #进行数据集划分：train|val|test
    (train_pair_set, val_pair_set, test_pair_set,
     positive_train_pair_set, negative_train_pair_set)= devide_data_set(positive_pair_set, negative_pair_set)

    #创建二进制文件
    create_all_data_file(train_pair_set, val_pair_set, test_pair_set, 
        positive_train_pair_set, negative_train_pair_set, FLAGS.data_dir)



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



def create_data_pair(group_data):
    '''
    根据分组组织好的数据创建样本数据对
    input:
        group_data:分组分好的数据
    output:
        positive_pair_set:同一个人的数据对集
        negative_pair_set:不是同一个人的数据对集
    '''
    positive_pair_set = []
    negative_pair_set = []

    for viewId in group_data.keys():
        #同一个视角下的同一个人，
        for personId in group_data[viewId].keys():
            positive_pair = _create_same_person_pair(group_data, viewId, persion)
            positive_pair_set.extend(positive_pair)

                    
        #同一个视角下的同一个人
        psersionID_list = list(group_data[viewId].keys())
        for i in range(len(psersionID_list)):
            for j in range(i+1, len(psersionID_list)):
              negative_pair =  _create_defferent_person_pair(group_data, viewId, psersionID_list[i], psersionID_list[j])
              negative_pair_set.extend(negative_pair)
            
    #保证各个视角的随机性进行重排序
    random.shuffle(positive_pair_set)
    random.shuffle(negative_pair_set)
    
    print("there are %d positive_pairs, %d negative_pairs" %(len(positive_pair_set), len(negative_pair_set)))
    
    return positive_pair_set, negative_pair_set
    


def _create_same_person_pair(group_data, viewId, personId):
    '''
    根据同一个人，将其下的两个camera的所有图片交叉组成positive_pair
    input:
        group_data: 分组数据结构
        viewId:     视角id
        personId:  人的id
    output:
        positive_pair_set: 组成的negative_pair结合

    '''
    positive_pair_set = []
    cameraId_list = list(group_data[viewId][persionId].keys())

     # 同一个相机下的同一个人
    for k in range(len(cameraId_list)):
        picture_list = group_data[viewId][personId][cameraId_list[k]]
        for i in range(len(picture_list) - 1):
            for j in range(len(picture_list)):
                picture1 = joint_id_to_name(viewId, personId, cameraId_list[k], picture_list[i])
                picture2 = joint_id_to_name(viewId, personId, cameraId_list[k], picture_list[j])
                positive_pair_set.extend((np.uint8(1), picture1, picture2))

    # 不同camera进行正组队
    for k in range(len(cameraId_list) - 1):
        for m in range(k+1, len(cameraId_list)):
            for picture1 in group_data[viewId][personId][cameraId_list[k]]:
                for pictureId2 in group_data[viewId][personId][cameraId_list[m]]:
                    picture1 = joint_id_to_name(viewId, personId, cameraId_list[k], pictureId1)
                    picture2 = joint_id_to_name(viewId, personId, cameraId_list[m], pictureId2)
                    positive_pair_set.extend((np.uint8(1), picture1, picture2))


    return positive_pair_set



def _create_defferent_person_pair(group_data, viewId, personId1, personId2):
    '''
    根据传入的两个不同人的id,将其下面的两个camera的所有图片，交叉组成negative_pair
    input:
        group_data: 分组数据结构
        viewId:     视角id
        personId1:  人1的id
        personId2:  人2的id
    output:
        negative_pair_set: 组成的negative_pair结合

    '''
    negative_pair_set = []

    for person1_cameraId in group_data[viewId][personId1].keys():
        for pictureId1 in group_data[viewId][personId1][personId1_cameraId]:
            for person2_cameraId  in group_data[viewId][persionId2][person2_cameraId]:
                picture1 = joint_id_to_name(viewId, personId1, person1_cameraId, pictureId1)
                picture2 = joint_id_to_name(viewId, personId2, person2_cameraId, pictureId2)
                negative_pair_set.extend((np.uint8(0), picture1, picture2))

    return negative_pair_set




def devide_data_set(positive_pair_set, negative_pair_set):
    '''
    根据正负样本对划分数据集
    input:
        positive_pair_set:同一个人的数据对集
        negative_pair_set:不是同一个人的数据对集
    output:
        train_pair_set:训练数据集(1/0, picture1, pituctrue2)
        val_pair_set:验证数据集(1/0, picture1, pituctrue2)
        test_pair_set:测试数据集(1/0, picture1, pituctrue2)
    '''
    
    #从正样本集中分割
    positive_test_pair_set = random.sample(positive_pair_set, int(0.1*len(positive_pair_set)))
    positive_train_pair_set = list(set(positive_pair_set).difference(set(positive_test_pair_set)))
    #positive_val_pair_set = random.sample(positive_train_pair_set, 480)
    #positive_train_pair_set = list(set(positive_train_pair_set).difference(set(positive_val_pair_set)))
    
    #从负样本集中分割
    # 根据论文对negative进行降采样
    negative_test_pair_set = random.sample(down_sample_negative_pair_set, int(0.1*len(positive_pair_set)*POSITIVE_NEGATIVE_RATE))
    negative_train_pair_set = list(set(negative_pair_set).difference(set(negative_test_pair_set)))
    #negative_val_pair_set = random.sample(negative_train_pair_set, int(0.1*len(positive_pair_set)*POSITIVE_NEGATIVE_RATE))
    #negative_train_pair_set = list(set(negative_train_pair_set).difference(set(negative_val_pair_set)))
    # 根据论文对negative进行将采样
    negative_train_pair_set_first_used = random.sample(negative_train_pair_set, int(POSITIVE_NEGATIVE_RATE* len(positive_train_pair_set)))
    
    #正负拼接
    train_pair_set = []
    train_pair_set.extend(positive_train_pair_set)
    train_pair_set.extend(negative_train_pair_set_first_used)
    random.shuffle(train_pair_set)
    val_pair_set = []
    #val_pair_set.extend(positive_val_pair_set)
    #val_pair_set.extend(negative_val_pair_set)
    #random.shuffle(train_pair_set)
    test_pair_set = []
    test_pair_set.extend(positive_test_pair_set)
    test_pair_set.extend(negative_test_pair_set)
    random.shuffle(test_pair_set)
    
    print('first_train_pair_set has %s samples' % len(train_pair_set))
    print('val_pair_set has %s samples' % len(val_pair_set))
    print('test_pair_set has %s samples' % len(test_pair_set))
    print('all positive_train_pair_set has %s samples' % len(positive_train_pair_set))
    print('all negative_train_pair_set has %s samples' % len(negative_train_pair_set))
    
    return train_pair_set, val_pair_set, test_pair_set, positive_train_pair_set, negative_train_pair_set



def create_data_name_file(data_set, data_dir, dest_name, file_limit=400):
    '''
    数据集中的文件名写入文件
    input:
        data_set:样本数据集
        data_dir:样本数据集目录，因为样本数据集中文件名没有路径信息
        dest_name:转换后生成的文件名前缀
        file_limit:转换后的文件大小限制，超过此文件大小将会自动分割,单位M
    '''
    serial_number = 1
    new_file_name = dest_name  + '_' + str(serial_number)+ ".txt"
    
    try:
        f = open(new_file_name, 'w')
        for data in data_set:
            f.write(str(data[0]))
            f.write(' ')
            f.write(os.path.join(data_dir, data[1]))
            f.write(' ')
            f.write(os.path.join(data_dir, data[2]))
            f.write('\n')
            f.flush()
            
            #一个图片对写完了判断文件大小,超过限制重新创建文件
            if os.path.getsize(new_file_name)/1024.0/1024.0 > file_limit:
                f.close()
                serial_number += 1
                new_file_name = dest_name  + '_' + str(serial_number)+ ".txt"
                f = open(new_file_name, 'w')
    finally:
        f.close()



def create_bin_data_file(data_set, data_dir, dest_name, file_limit=400):
    '''
    数据集中的将文件转换成二进制文件
    input:
        data_set:样本数据集
        data_dir:样本数据集目录，因为样本数据集中文件名没有路径信息
        dest_name:转换后生成的文件名前缀
        file_limit:转换后的文件大小限制，超过此文件大小将会自动分割,单位M
    '''
    serial_number = 1
    new_file_name = dest_name  + '_' + str(serial_number)+ ".bin"
    
    try:
        f = open(new_file_name, 'wb')
        for data in data_set:
            f.write(data[0])
            f.write(turn_picture_to_bin(os.path.join(data_dir, data[1])))
            f.write(turn_picture_to_bin(os.path.join(data_dir, data[2])))
            f.flush()
            
            #一个图片对写完了判断文件大小,超过限制重新创建文件
            if os.path.getsize(new_file_name)/1024.0/1024.0 > file_limit:
                f.close()
                serial_number += 1
                new_file_name = dest_name  + '_' + str(serial_number)+ ".bin"
                f = open(new_file_name, 'wb')
    finally:
        f.close()
    
    
    
def create_all_data_file(train_pair_set, val_pair_set, test_pair_set,
                        positive_train_pair_set, negative_train_pair_set, data_dir):
    '''
    数创建所有据集中的二进制文件
    input:
        train_pair_set:训练数据集(1/0, picture1, pituctrue2)
        val_pair_set:验证数据集(1/0, picture1, pituctrue2)
        test_pair_set:测试数据集(1/0, picture1, pituctrue2)
        positive_train_pair_set:所有正样本集合
        negative_train_pair_set:所有负样本集合
        data_dir:样本数据集目录，因为样本数据集中文件名没有路径信息
    '''
    create_bin_data_file(train_pair_set, data_dir, 'train_set')
    create_bin_data_file(val_pair_set, data_dir, 'val_set')
    create_bin_data_file(test_pair_set, data_dir, 'test_set')

    create_data_name_file(positive_train_pair_set, data_dir, 'positive_train_set')
    create_data_name_file(negative_train_pair_set, data_dir, 'negative_train_set')
    print('data prepare done!')



def read_file_name_to_bin(file_name, data_dir):
    '''
    读取存储匹配成对的文件并转换成二进制文件
    input:
        file_name:存储样本对的文件名
        data_dir:样本数目数据集目录，因为样本数据集中文件名没有路径信息

    '''
    data_pair = []
    with open(file_name) as file:
        for line in file:
            line = line.strip('\n')
            label, picture1, picture2 = line.split('' )
            data_dir.append((np.uint8(int(label)), picture1, picture2))

    name, _ = os.path.join(file_name)
    create_bin_data_file(data_pair, data_dir, name) 



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
    
    return np.array(l)




def joint_id_to_name(viewId, personId, cameraId, pictureId):
    '''
    将路径和各种id拼接成全路径名
    '''
    file_name =  viewId + '_' + personId + '_' + cameraId + '_' + pictureId + '.png'
    return  file_name



def get_files_by_name_regular(data_dir, name_regular):
    '''
    通过命名规则制定目录中获取相应的文件列表
    input:
        data_dir:指定目录
        name_regular:文件命名规则
    output:
        文件列表
    '''
    files = [files for files in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, files)) ]
    filenames = [f for f in files if fnmatch(f, name_regular)]
    


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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str,
                  default='./cuhk03_labeled',
                  help='Directory for input data')
    
        
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)