# -*- coding: utf-8 -*-
#===============================
#模型构建程序：
#    按照论文结构构建模型
#===============================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

FLAGS = tf.app.flags.FLAGS
TOWER_NAME = 'tower'
NUM_CLASSES = 2

# Basic model parameters.
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('batch_size', 150,
                            """Number of images to process in a batch.""")#128应该更合适
tf.app.flags.DEFINE_float('weiht_decay', 0.0005,
                            """value of the weiht decay.""")#论文值


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))
  
  
def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var



def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

      Note that the Variable is initialized with a truncated normal distribution.
      A weight decay is added only if one is specified.

      Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var



def _self_extend(input_x, shape):
    '''
    将input_x进行扩充，
    将input_x中的height, width两个维度的每个元素扩展为shape的矩阵，矩阵的值全部为input_x对应位置的元素，反卷积操作
    是一个求平均的反操作
    '''
    height, width = shape
    kernel_array = np.zeros((height, width, 25, input_x.shape[3]))
    for i in range(input_x.shape[3]):
        kernel_array[:,:,i,i] = 1.0/height/width
    
    kernel = tf.constant(kernel_array, dtype=tf.float32, shape=[height, width, 25, input_x.shape[3]])
    extended_input = tf.nn.conv2d_transpose(input_x,
                                           kernel, 
                                           [input_x.shape[0].value, input_x.shape[1].value*height, input_x.shape[2].value*width, 25],
                                           [1,5,5,1],
                                           padding = 'VALID')
    return extended_input



def _neighbor_extend(input_x, shape):
    '''
    将input_x进行扩充，
    对input_x中的height, width 的每个元素，以其为中心，截取shape的小矩阵，然后再挨个拼接起来，扩展成大矩阵
    '''  
    height, width = shape
    extended_input = None
    extended_input_row = None
    input_x_pad = tf.pad(input_x, [[0,0],[2,2],[2,2],[0,0]])#将边缘填零
    for i in range(2, input_x.shape[1].value+2):
        for j in range(2, input_x.shape[2].value+2):
            if extended_input_row == None:
                extended_input_row = input_x_pad[:,i-2:i+3,j-2:j+3,:]
            else:
                extended_input_row = tf.concat([extended_input_row, input_x_pad[:,i-2:i+3,j-2:j+3,:]], 2)#master版本中参数顺序变化了
        if extended_input == None:
            extended_input = extended_input_row
        else:
            extended_input = tf.concat([extended_input, extended_input_row], 1)#master版本中参数顺序变化了
        
        extended_input_row = None
        
    return  extended_input      

            
    
def inference(images):
    '''
    模型定义
    input:
        images:数据读取模块的输入
    output:
        Logits.
    '''
    images1, images2 = tf.split(images, num_or_size_splits = 2, axis = 3)#切分出来,master版本参数顺序改变了
    
    #----L1
    with tf.variable_scope('Tied_ConvMax_Pooling1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 20],
                                         stddev=5e-2,
                                         wd=FLAGS.weiht_decay)
        biases = _variable_on_cpu('biases', [20], tf.constant_initializer(0.0))
        
        conv1_1 = tf.nn.conv2d(images1, kernel, [1,1,1,1], padding='VALID')
        pre_activation1_1 = tf.nn.bias_add(conv1_1, biases)
        active1_1 = tf.nn.relu(pre_activation1_1, name=scope.name)
        _activation_summary(active1_1)
        
        conv1_2 = tf.nn.conv2d(images2, kernel, [1,1,1,1], padding='VALID')
        pre_activation1_2 = tf.nn.bias_add(conv1_2, biases)
        active1_2 = tf.nn.relu(pre_activation1_2, name=scope.name)
        _activation_summary(active1_2)
        
    pool1_1 = tf.nn.max_pool(active1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool1_1')
    pool1_2 = tf.nn.max_pool(active1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool1_2')
    #-----L2
    with tf.variable_scope('Tied_ConvMax_Pooling2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 20, 25],
                                         stddev=5e-2,
                                         wd=FLAGS.weiht_decay)
        biases = _variable_on_cpu('biases', [25], tf.constant_initializer(0.0))
        
        conv2_1 = tf.nn.conv2d(pool1_1, kernel, [1,1,1,1], padding='VALID')
        pre_activation2_1 = tf.nn.bias_add(conv2_1, biases)
        active2_1 = tf.nn.relu(pre_activation2_1, name=scope.name)
        _activation_summary(active2_1)
        
        conv2_2 = tf.nn.conv2d(pool1_2, kernel, [1,1,1,1], padding='VALID')
        pre_activation2_2 = tf.nn.bias_add(conv2_2, biases)
        active2_2 = tf.nn.relu(pre_activation2_2, name=scope.name)
        _activation_summary(active2_2)
        
    pool2_1 = tf.nn.max_pool(active2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool2_1')
    pool2_2 = tf.nn.max_pool(active2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool2_2')
    #------L3
    with tf.variable_scope('Cross_Input_Neighborhood_Differences') as scope:
        K1 =  tf.subtract( _self_extend(pool2_1, (5, 5)) , _neighbor_extend(pool2_2, (5, 5)))
        K2 =  tf.subtract(_self_extend(pool2_2, (5, 5)) , _neighbor_extend(pool2_1, (5, 5)))
    #------L4
    with tf.variable_scope('Patch_Summary_Features') as scope:
        kernel1 = _variable_with_weight_decay('weights1',
                                         shape=[5, 5, 25, 25],
                                         stddev=5e-2,
                                         wd=FLAGS.weiht_decay)
        biases1 = _variable_on_cpu('biases1', [25], tf.constant_initializer(0.0))
        conv4_1 = tf.nn.conv2d(K1, kernel1, [1,5,5,1], padding='VALID')
        pre_activation4_1 = tf.nn.bias_add(conv4_1, biases1)
        active4_1 = tf.nn.relu(pre_activation4_1, name=scope.name)
        _activation_summary(active4_1)
        
        kernel2 = _variable_with_weight_decay('weights2',
                                         shape=[5, 5, 25, 25],
                                         stddev=5e-2,
                                         wd=FLAGS.weiht_decay)
        biases2 = _variable_on_cpu('biases2', [25], tf.constant_initializer(0.0))
        conv4_2 = tf.nn.conv2d(K2, kernel2, [1,5,5,1], padding='VALID')
        pre_activation4_2 = tf.nn.bias_add(conv4_2, biases2)
        active4_2 = tf.nn.relu(pre_activation4_2, name=scope.name)
        _activation_summary(active4_2)
    #------L5
    with tf.variable_scope('Across_Patch_Features') as scope:
        kernel1 = _variable_with_weight_decay('weights1',
                                         shape=[3, 3, 25, 25],
                                         stddev=5e-2,
                                         wd=FLAGS.weiht_decay)
        biases1 = _variable_on_cpu('biases1', [25], tf.constant_initializer(0.0))
        conv5_1 = tf.nn.conv2d(active4_1, kernel1, [1,1,1,1], padding='SAME')
        pre_activation5_1 = tf.nn.bias_add(conv5_1, biases1)
        active5_1 = tf.nn.relu(pre_activation5_1, name=scope.name)
        _activation_summary(active5_1)
        
        kernel2 = _variable_with_weight_decay('weights2',
                                         shape=[3, 3, 25, 25],
                                         stddev=5e-2,
                                         wd=FLAGS.weiht_decay)
        biases2 = _variable_on_cpu('biases2', [25], tf.constant_initializer(0.0))
        conv5_2 = tf.nn.conv2d(active4_2, kernel2, [1,1,1,1], padding='SAME')
        pre_activation5_2 = tf.nn.bias_add(conv5_2, biases2)
        active5_2 = tf.nn.relu(pre_activation5_2, name=scope.name)
        _activation_summary(active4_2)
    pool5_1 = tf.nn.max_pool(active5_1, ksize=[1, 3, 4, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool5_1')
    pool5_2 = tf.nn.max_pool(active5_2, ksize=[1, 3, 4, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool5_2')
    #------L6，全连接层
    with tf.variable_scope('Higher_Order_Relationships') as scope:
        M = tf.concat([pool5_1, pool5_2], 3)#合并两个图片,master版本中参数顺序颠倒了
        reshape = tf.reshape(M, [reshape.get_shape()[1].value, -1])#不能写成FLAGS.batch_size,否则在做deploy时，输入数据数就会被固定死
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 500],
                                          stddev=0.04, wd=FLAGS.weiht_decay)
        biases = _variable_on_cpu('biases', [500], tf.constant_initializer(0.1))
        local6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local6)
        
    # output
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', shape=[500, NUM_CLASSES],
                                          stddev=1/500, wd=FLAGS.weiht_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local6, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
      
    return softmax_linear
    


def loss(logits, labels):
    """
    Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
          of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
  
    # 计算一个mini_batch的平均loss
  
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
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


