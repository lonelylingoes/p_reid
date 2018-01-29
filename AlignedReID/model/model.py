#-*- conding:utf-8 -*-
#====================
#模型构建程序
#====================

from cnn_net import resnet_model
from cnn_net import resnetX_model
from utils import read_conf


resnet_size = int(read_conf.get_conf_param('cnn_param', 'resnet_size'))
data_format = read_conf.get_conf_param('cnn_param', 'data_format')



def global_feature(inputs, data_format):
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
  block_fn = resnet_model.get_model_param(resnet_size)['block']
  global_feature = tf.reshape(inputs,
                        [-1, 512 if block_fn is building_block else 2048])
  global_feature = tf.identity(inputs, 'global_feature')

  return global_feature



def local_feautre(inputs, data_format):
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
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)#再用一个1x1的卷积，将通道缩小
  local_feature = tf.identity(inputs, 'local_feature')
    
  return local_feature



def classfication_feature(inputs, num_classes):
  '''
  分类分支输出的特征向量
  Args:
    inputs: 全局特征向量  
    num_classes: The number of possible classes for identities(即一个batch中行人个数).

  Returns:
    分类的logits
  '''
  inputs = tf.layers.dense(inputs=inputs, units=num_classes)
  logits = tf.identity(inputs, 'classfication_logits')

  return logits



def global_distance(global_feature):
  '''
  计算全局距离
  '''



def local_distance(local_feature):
  '''
  计算局部距离
  '''












 def get_resnet_feature_map(inputs):
 	'''
	由resnet获取feature_map
 	'''
 	network = resnet_model.imagenet_resnet_v2(resnet_size, data_format)
  	feature_map = network(inputs=inputs, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  	return feature_map




 def get_resnet_X_feature_map(inputs):
 	'''
	由resnet_X获取feature_map
 	'''
 	network = resnet_model.imagenet_resnetX_v2(resnet_size, data_format)
  	feature_map = network(inputs=inputs, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  	return feature_map