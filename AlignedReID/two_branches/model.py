#-*- coding:utf-8 -*-
#====================
#模型构建程序
#====================
import sys
sys.path.append('../')

from cnn_net import resnet_model
from cnn_net import resnetX_model
from utils import read_conf
import tensorflow as tf
import numpy as np


RESNET_SIZE = read_conf.get_conf_int_param('cnn_param', 'resnet_size')
DATA_FORMAT = read_conf.get_conf_str_param('cnn_param', 'data_format')

IMAGES_PER_ID= read_conf.get_conf_int_param('hyper_param', 'images_per_id')#每个人K张图片
TRIPLET_PER_BATCH = read_conf.get_conf_int_param('hyper_param', 'triplet_per_batch')#一个batich中有B个triplet
ID_PER_BATCH = 3*TRIPLET_PER_BATCH/IMAGES_PER_ID#由3*B = P*K 可以算到一个batch中含有P=3*B/K个人

LOCAL_MARGIN = read_conf.get_conf_float_param('hyper_param', 'local_margin')
GLOBAL_MARGIN = read_conf.get_conf_float_param('hyper_param', 'global_margin')
MUTUL_CLASSIFICATION_WEIGHT = read_conf.get_conf_float_param('hyper_param', 'mutul_classification_weight')
MUTUL_METRIC_WEIGHT = read_conf.get_conf_float_param('hyper_param', 'mutul_metric_weight')
WEIGHT_DECAY = read_conf.get_conf_float_param('hyper_param', 'wd')
IDENTITIES = read_conf.get_conf_int_param('hyper_param', 'identities')



def get_global_feature(inputs, data_format, name):
    '''
    获取全局特征向量  
    args:
      inputs:cnn提取的feature map
      data_format:数据格式是[h,w,c]还是[c,w,h]
    output:
        全局特征
    '''
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=7, strides=1, padding='VALID',
        data_format=data_format)
    #global_features = tf.reshape(inputs, [-1, 2048])
    global_features = tf.identity(inputs, name)
    #print('=====global features shape is:', global_features.shape)

    return global_features




def get_local_feature(inputs, data_format, name):
    '''
    获取特征向量
    args:
        inputs:cnn提取的feature map
        data_format:数据格式是[h,w,c]还是[c,w,h]
    output:
        local_feature:局部特征
    '''
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=[1, 7], strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.layers.conv2d(
    inputs=inputs, filters=128, kernel_size=1, strides=1,
        padding='VALID', use_bias=False,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
       data_format=data_format)#再用一个1x1的卷积，将通道缩小
    local_features = tf.identity(inputs, name)
    #print('=====local features shape is:', local_features.shape)
    
    return local_features



def get_classfication_feature(inputs, num_classes, name):
    '''
    分类分支输出的特征向量
    Args:
        inputs: 全局特征向量  
        num_classes: The number of possible classes for identities(即一个batch中行人个数).
        name:用来区分是resnet还是resnetX分支
    Returns:
        分类的logits
    '''
    inputs = tf.squeeze(inputs)
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    logits = tf.identity(inputs, name)

    return logits



def get_global_dis_mat(global_features):
    '''
    计算全局距离
    output:
      NxN的距离值矩阵
    '''
    size = global_features.shape[0].value
    global_features = tf.squeeze(global_features)
    distance_matrix = []#不能对一个tensor的元素直接赋值，只能采取这种方式
    #这个矩阵是对称阵，对角线元素是0
    # 不能挨个元素循环，效率太低了，采用向量方式计算
    for i in range(size):
          dis_tensor = tf.sqrt(tf.reduce_sum(tf.square(global_features[i,:]-global_features), axis = 1))
          distance_matrix.append(dis_tensor)

    distance_matrix = tf.stack(distance_matrix, 0)
    tf.assert_equal(distance_matrix, tf.transpose(distance_matrix))
    #print('global distance_matrix shape', distance_matrix)
    return distance_matrix



def get_local_distance(local_feature1, local_feature2, data_format):
    '''
    获取局部距离
    '''
    return cal_local_shortest_distance(get_local_distance_mat(local_feature1, local_feature2, data_format))



def get_local_distance_mat(local_feature1, local_feature2, data_format):
    '''
    计算两个特征向量之间的局部距离矩阵
    '''
    distance_matrix = []
    
    # 先计算距离矩阵
    if data_format == 'channels_first':
        dim1 = local_feature1.shape[1].value
        dim2 = local_feature2.shape[1].value
        # 两两间的距离矩阵
        local_feature1 = tf.squeeze(local_feature1)#(128,7)
        local_feature2 = tf.squeeze(local_feature2)#(128,7)
        
        for i in range(dim1):
            for j in range(dim2):
                dis_tensor = tf.sqrt(tf.reduce_sum(tf.square(local_feature1[:,i]-local_feature2[:,j])))
                e_dis = tf.exp(dis_tensor)
                distance_matrix.append(tf.div(e_dis - 1, e_dis + 1))
    else:
        dim1 = local_feature1.shape[0].value
        dim2 = local_feature2.shape[0].value
        # 两两间的距离矩阵
        local_feature1 = tf.squeeze(local_feature1)#(7,128)
        local_feature2 = tf.squeeze(local_feature2)#(7,128)

        for i in range(dim1):
            for j in range(dim2):
                dis_tensor = tf.sqrt(tf.reduce_sum(tf.square(local_feature1[i, :]-local_feature2[j, :])))
                e_dis = tf.exp(dis_tensor)
                distance_matrix.append(tf.div(e_dis - 1, e_dis + 1))
    
    distance_matrix = tf.stack(distance_matrix, 0)
    distance_matrix = tf.reshape(distance_matrix, [dim1, dim2])
    #print('local distance_matrix shape', distance_matrix.shape)
    return distance_matrix



def cal_local_shortest_distance(distance_matrix):
    '''
    阵元素中的最短路径距离，直观上看，该问题非常复杂，找到最短距离需要大量遍历。
    所以论文中采用了动态规划的方法：
    每一个位置mat[i][j]只可能来自mat[i][j-1]向右走一个结点或者mat[i-1][j]向下走一个节点，
    因此只需要比较到达mat[i][j-1]和到达mat[i-1][j]的路径较小值加上mat[i][j]就是所求答案，即；
    S[i][j] = min(S[i-1][j], S[i][j-2]) + mat[i][j]
    思路：求出到达每一个结点mat[i][j]的最小路径将其保存在数组dis[i][j]中，
        求任意dis[i][j]的值完全依赖于dis[i-1][j]和dis[i][j-1]，
        因此先求出dis[][]数组的第1行和第1列，然后从上到下，从左到右计算出每一个位置的结果值。
    ==================================================
    '''
    row_num = distance_matrix.shape[0].value
    column_num = distance_matrix.shape[1].value
    
    #第一个元素就是值
    #shortest_dis_mat = tf.zeros_like(distance_matrix)#只是用来理解用，该方式无法运行
    #shortest_dis_mat[0][0] = distance_matrix[0][0]
    shortest_dis_mat = None
    #求解第1列的结果
    first_column = []
    first_column.append(distance_matrix[0][0])
    for i in range(1, row_num):
        #shortest_dis_mat[i][0] = shortest_dis_mat[i-1][0] + distance_matrix[i][0]
        first_column.append(first_column[i-1] + distance_matrix[i][0])
    first_column = tf.stack(first_column, 0)
    first_column = tf.reshape(first_column, [row_num, 1])
        
    #求解第1行的结果
    first_row = []
    first_row.append(distance_matrix[0][0])
    for i in range(1, column_num):
        #shortest_dis_mat[0][i] = shortest_dis_mat[0][i-1] + distance_matrix[0][i]
        first_row.append(first_row[i-1] + distance_matrix[0][i])
    first_row = tf.stack(first_row, 0)
    first_row = tf.reshape(first_row, [1, column_num])
        
    # ========================================================
    #  从左到右，从上到下扩展，也就是先扩展成一行，再将行拼接起来
    #
    #  第一行和第一列已经有了，通过连接的方式将其他也连接起来
    #  * * * * * * *   
    #  * - - - - - -
    #  * - - - - - -
    #  * - - - - - -
    #  * - - - - - -
    #  * - - - - - -
    #  * - - - - - -
    # ===========================================================
    shortest_dis_mat = first_row
    for i in range(1, row_num):
        row = []
        row.append(first_column[i][0])#将第一列的元素先加到一行的首位
        for j in range(1, column_num):
            #shortest_dis[i][j] = distance_matrix[i][j] + tf.minimum(shortest_dis_mat[i-1][j], shortest_dis_mat[i][j-1])
            row_last = shortest_dis_mat[i-1][j]
            cloumn_last = row[j-1]
            row.append(distance_matrix[i][j]  + tf.minimum(row_last, cloumn_last))
            
        row = tf.stack(row, 0)
        row = tf.reshape(row, [1, column_num])
        shortest_dis_mat = tf.concat([shortest_dis_mat, row], 0)
            
    #print('shortest_dis_mat shape;', shortest_dis_mat.shape)
            
    # 最右下角那个元素就是最终结果
    return shortest_dis_mat[row_num-1][column_num-1]
  



def get_resnet_feature_map(inputs, resnet_size, mode, data_format):
    '''
    由resnet获取feature_map
    args:
        mode:标志当前模式，此处使用方式可能有问题
    '''
    network = resnet_model.imagenet_resnet_v2(resnet_size, data_format)
    feature_map = network(inputs=inputs, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    return feature_map




def get_resnetX_feature_map(inputs, resnet_size, mode, data_format):
    '''
    由resnet_X获取feature_map
    args:
        mode:标志当前模式，此处使用方式可能有问题
    '''
    network = resnetX_model.imagenet_resnetX_v2(resnet_size, data_format)
    feature_map = network(inputs=inputs, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    return feature_map



def trihard_loss(global_dis_mat, local_features, images_per_id,
                 global_margin, local_margin, data_format):
    '''
    根据(全局)距离找到hardness triplet 同时计算Loss
    inputs:
        
    outputs:
        为了后面能够使用到，将全局距离矩阵也返回出来
    '''
    size = global_dis_mat.shape[0].value
    hard_loss = 0

    for i in range(size):
        index_of_person = i // images_per_id#由当前i得到，当前所属的人的索引
        per_image_dist_mat = global_dis_mat[:,i]#第i个图片与每个图片计算的距离列表
   
        # 取连续在一起的一个人的几张图片
        same_person = per_image_dist_mat[index_of_person*images_per_id : (index_of_person+1)*images_per_id]
        # tensor需要这样切分
        hardest_positive_index = tf.argmax(same_person)#该返回值为int64
        hardest_positive_index = tf.cast(hardest_positive_index, tf.int32)#作为tensor下标索引只支持int32
        hardest_positive_index += index_of_person * images_per_id#上面算得的是切片后的索引，需要转换成最初的索引
        
        # 剔除掉连续在一起的一个人的几张图片，再将前后两个部分链接在一起
        first_part = per_image_dist_mat[0 : index_of_person*images_per_id]
        last_part = per_image_dist_mat[(index_of_person+1)*images_per_id : size]
        different_person = tf.concat([first_part, last_part], 0)
        hardest_negative_index = tf.argmin(different_person)#该返回值为int64
        hardest_negative_index = tf.cast(hardest_negative_index, tf.int32)#作为tensor下标索引只支持int32
        #如果算出来的index在前一部分则就是原来矩阵中的数值，如果不是则需要加上剔除掉的数目
        #不能将一个tensor放到条件语句中直接比较，需要用tf.cond来判断
        hardest_negative_index = tf.cond(hardest_negative_index >= index_of_person*images_per_id,
                                         lambda:tf.add(hardest_negative_index, images_per_id),
                                         lambda:hardest_negative_index)
        
        # 论文的意思是分别用margin,算两个loss的和，
        hard_positive_global_dis = global_dis_mat[i][hardest_positive_index]
        hard_negative_global_dis = global_dis_mat[i][hardest_negative_index]
        hard_positive_local_dis = get_local_distance(local_features[i], local_features[hardest_positive_index], data_format)
        hard_negative_local_dis = get_local_distance(local_features[i], local_features[hardest_positive_index], data_format)
        hard_loss += tf.maximum(0.0, 
                                global_margin + hard_positive_global_dis - hard_negative_global_dis + \
                                local_margin + hard_positive_local_dis - hard_negative_local_dis)
        print('===the %dth pictures' % (i+1))
    hard_loss /= size#求平均
    
    return hard_loss


def resnet_inference(images, mode):
    '''
    resnet的前向推理
    args:
        images:图片
        mode:是否为训练或测试，tf.estimator.ModeKeys.TRAIN
    returns:
        feature_map和global_feature
    '''
    with tf.variable_scope('resnet') as scope:
        #计算feature_map
        resnet_feature_map = get_resnet_feature_map(images, RESNET_SIZE, mode, DATA_FORMAT)
        resnet_global_feaures = get_global_feature(resnet_feature_map, DATA_FORMAT, 'resnet_global_feaures')
        
        return resnet_feature_map, resnet_global_feaures



def resnetX_inference(images, mode):
    '''
    resnet的前向推理
    args:
        images:图片
        mode:是否为训练或测试，tf.estimator.ModeKeys.TRAIN
    returns:
        feature_map和global_feature
    '''
    with tf.variable_scope('resneXt') as scope:
        #计算feature_map
        resnetX_feature_map = get_resnetX_feature_map(images, RESNET_SIZE, mode, DATA_FORMAT)
        resnetX_global_feaures = get_global_feature(resnetX_feature_map, DATA_FORMAT, 'resnetX_global_feaures')
        
        return resnetX_feature_map, resnetX_global_feaures



def loss(labels, 
         resnet_feature_map, resnet_global_feaures,
         resnetX_feature_map, resnetX_global_feaures):
    '''
    计算整个模型的Loss
    args:
        labels:分类的标签

    returns:
        totaol_loss
        
    '''
    resnet_local_feaures = get_local_feature(resnet_feature_map, DATA_FORMAT, 'restnet_local_feature')
    resnet_global_dis_mat = get_global_dis_mat(resnet_global_feaures)
    resnetX_local_feaures = get_local_feature(resnetX_feature_map, DATA_FORMAT, 'restnetX_local_feature')
    resnetX_global_dis_mat = get_global_dis_mat(resnetX_global_feaures)
    
    #==========================
    #===计算metric mutual loss===
    #==========================
    ZG_resnet_global_dis_mat = tf.constant(resnet_global_dis_mat)
    ZG_resnetX_global_dis_mat = tf.constant(resnetX_global_dis_mat)
    metric_mutual_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(ZG_resnet_global_dis_mat-resnetX_global_dis_mat)) + \
                tf.reduce_sum(tf.square(ZG_resnetX_global_dis_mat-resnet_global_dis_mat))) 
    tf.add_to_collection('losses', metric_mutual_loss * MUTUL_METRIC_WEIGHT)
    
    #==========================
    #===restnet这一分支的loss===
    #==========================
    #计算classfication loss
    resnet_logits = get_classfication_feature(resnet_global_feaures, IDENTITIES, 'resnet_logits')
    resnet_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=resnet_logits, name='resnet_cross_entropy_per_example')#有可能有问题
    rensnet_cross_entropy_mean = tf.reduce_mean(resnet_cross_entropy, name='resnet_cross_entropy')
    tf.add_to_collection('losses', rensnet_cross_entropy_mean)
    
    # 计算metric loss
    resnet_l2_loss = WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() 
            if 'batch_normalization' not in v.name and  'resnet' in v.name])
    tf.add_to_collection('losses', resnet_l2_loss)
    resnet_trihard_loss = trihard_loss(resnet_global_dis_mat, resnet_local_feaures,
                                                          IMAGES_PER_ID, GLOBAL_MARGIN, LOCAL_MARGIN, DATA_FORMAT)
    tf.add_to_collection('losses', resnet_trihard_loss)
    print('===resnet_trihard_loss done!')
    
    #==========================
    #===restnetX这一分支的loss===
    #==========================
    #计算classfication loss
    resnetX_logits = get_classfication_feature(resnetX_global_feaures, IDENTITIES, 'resnetX_logits')
    resnetX_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=resnetX_logits, name='resnetX_cross_entropy_per_example')#有可能有问题
    rensnetX_cross_entropy_mean = tf.reduce_mean(resnetX_cross_entropy, name='resnetX_cross_entropy')
    tf.add_to_collection('losses', rensnetX_cross_entropy_mean)
    
    # 计算metric loss
    resnetX_l2_loss = WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() 
            if 'batch_normalization' not in v.name and 'resneXt' in v.name])#要将上面添加的变量区排除掉
    tf.add_to_collection('losses', resnetX_l2_loss)
    resnetX_trihard_loss = trihard_loss(resnetX_global_dis_mat, resnetX_local_feaures,
                                                    IMAGES_PER_ID, GLOBAL_MARGIN, LOCAL_MARGIN, DATA_FORMAT)
    tf.add_to_collection('losses', resnetX_trihard_loss)
    print('===resnetX_trihard_loss done!')
    
    #==========================
    #===计算classfication mutual loss===
    #==========================
    KL_loss1 = tf.reduce_mean(tf.reduce_sum(tf.softmax(resnet_logits) * tf.log(tf.softmax(resnet_logits)/ tf.softmax(resnetX_logits))))
    tf.add_to_collection('losses', KL_loss1 * MUTUL_CLASSIFICATION_WEIGHT)
    KL_loss2 = tf.reduce_mean(tf.reduce_sum(tf.softmax(resnetX_logits) * tf.log(tf.softmax(resnetX_logits)/ tf.softmax(resnet_logits))))
    tf.add_to_collection('losses', KL_loss2 * MUTUL_CLASSIFICATION_WEIGHT)


    return tf.add_n(tf.get_collection('losses'), name='total_loss')



def add_loss_summaries(total_loss):  
    '''
    通过使用指数衰减，来维护变量的滑动均值。    
    滑动均值是通过指数衰减计算得到的。shadow variable的初始化值和trained variables相同，其更新公式为:  
    shadow_variable = decay * shadow_variable + (1 - decay) * variable
    
    当训练模型时，维护训练参数的滑动均值是有好处的。
    在测试过程中使用滑动参数比最终训练的参数值本身，会提高模型的实际性能（准确率）。
    apply()方法会添加trained variables的shadow copies，并添加操作来维护变量的滑动均值到shadow copies。
    average 方法可以访问shadow variables，在创建evaluation model时非常有用。  
    '''
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg') #创建一个新的指数滑动均值对象  
    losses = tf.get_collection('losses')# 从字典集合中返回关键字'losses'对应的所有变量，包括交叉熵损失和正则项损失  
    # 创建‘shadow variables’,并添加维护滑动均值的操作  
    loss_averages_op = loss_averages.apply(losses + [total_loss])#维护变量的滑动均值，返回一个能够更新shadow variables的操作
    
    for l in losses+[total_loss]:  
        tf.summary.scalar(l.op.name+'_raw', l) #保存变量到Summary缓存对象，以便写入到文件中  
        tf.summary.scalar(l.op.name, loss_averages.average(l)) 
    return loss_averages_op  #返回损失变量的更新操作 
