# -*- coding: utf-8 -*-
#===============================
#模型训练程序
#    
#===============================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import data_input
import model


from utils import read_conf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('ckpt_dir', '../ckpt_dir',
                           """Directory where to write event logs and checkpoint.""")

#输入目录
INPUT_DIR = ['../../data_set/CUHK03/cuhk03_labeled/train/bin_file',
             '../../data_set/Market1501/bounding_box_train/bin_file',
             '../../data_set/CUHK-SYSU/dataset/Image/train/bin_file']


# Constants describing the training process.
MOVING_AVERAGE_DECAY = read_conf.get_conf_float_param('hyper_param', 'moving_average_decay')# The decay to use for the moving average.
INITIAL_LEARNING_RATE = read_conf.get_conf_float_param('hyper_param', 'inital_learning_rate')# Initial learning rate.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = read_conf.get_conf_int_param('hyper_param', 'pictures_per_epoch')#每个epoch中样本数
TRIPLET_PER_BATCH = read_conf.get_conf_int_param('hyper_param', 'triplet_per_batch')#一个batch中triple个数
BATCHES_PER_EPOCH = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // (3*TRIPLET_PER_BATCH)#一个epoch中batch 个数
MAX_EPOCH = read_conf.get_conf_int_param('hyper_param', 'max_epoch')#最大的epoch数
MAX_STEP = MAX_EPOCH * BATCHES_PER_EPOCH # 最大训练步数
DATA_FORMAT = read_conf.get_conf_str_param('cnn_param', 'data_format')#图片格式

INITIAL_LEARNING_RATE = 1e-3

def _inverse_time_decay(learning_rate, global_step, name=None):
    """
    论文中的学习率衰减的计算公式，tensorflow中没有提供，需要自己实现
    
      Args:
          learning_rate: A scalar `float32` or `float64` `Tensor` or a
              Python number.  The initial learning rate.
          global_step: A Python number.
              Global step to use for the decay computation.  Must not be negative.
          decay_rate: A Python number.  The decay rate.
          exponent_rate:   根据论文公式，为指数p
          staircase: Whether to apply decay in a discrete staircase, as opposed to
              continuous, fashion.
          name: String.  Optional name of the operation.  Defaults to
              'InverseTimeDecay'.
    
      Returns:
          A scalar `Tensor` of the same type as `learning_rate`.  The decayed
              learning rate.
    
      Raises:
          ValueError: if `global_step` is not supplied.
    """
    if global_step is None:
        raise ValueError("global_step is required for inverse_time_decay.")
        
    
    with ops.name_scope(name, "TimeDecay",
              [learning_rate, global_step]) as name:
        current_epoch = global_step / BATCHES_PER_EPOCH 

# =============================================================================
#         if current_epoch < 80:
#             lr = learning_rate
#         elif current_epoch >= 80 and current_epoch< 160:
#             lr = 1e-4
#         elif current_epoch >= 160:
#             lr = 1e-5
# =============================================================================
        f = lambda:tf.cond(tf.logical_and(current_epoch >= 80 , current_epoch< 160),
                             lambda:tf.constant(1e-4),
                             lambda:tf.constant(1e-5))
        lr = tf.cond(current_epoch < 80, lambda:tf.constant(learning_rate), f)
        return lr



def one_step_train(total_loss, global_step):
    '''
    Train model.
      Create an optimizer and apply to all trainable variables. Add moving
      average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    '''
    #计算衰减过后的学习率
    lr = _inverse_time_decay(INITIAL_LEARNING_RATE, global_step)
    tf.summary.scalar('learning_rate', lr)
    
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = model.add_loss_summaries(total_loss)
    
    #计算梯度
    #直接调用minimize()相当于调用了compute_gradients()和apply_gradients()
    #为了在apply_gradients()之前对梯度做一些处理，需要分开写
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)#论文中值
        grads = opt.compute_gradients(total_loss)
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
   
    # 为可训练变量添加柱状图.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    
    # 为梯度添加柱状图.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
           
    # 跟踪每个可训练变量的移动平均值
    variable_averages = tf.train.ExponentialMovingAverage(
           MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    #控制计算流图，给图中的计算指定顺序
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')# Does nothing. Only useful as a placeholder for control edges.

    return train_op
   


def train_from_start(ckpt_dir, max_step):
    '''
    判断是否从头开始训练
    input:
        ckpt_dir:存盘点目录
        max_step:指定的最大训练步数，用于判断是否已经训练完
    '''   
    ckpt = tf.train.get_checkpoint_state(ckpt_dir) # 通过检查点文件锁定最新的模型
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        if int(global_step) < max_step - 1:
            return False
    return True
    
    

def train():
    '''
    模型训练函数
    '''
    #指定当前图为默认graph  
    with tf.Graph().as_default():
        # 强迫从cpu来读取数据，防止跑到GPU上变慢.
        with tf.device('/cpu:0'):
            images, labels = data_input.train_inputs(INPUT_DIR, TRIPLET_PER_BATCH*3)
        
        #训练步数起始值
        start_step = 1
        # 前向
        resnet_feature_map, resnet_global_feaures = model.resnet_inference(images, tf.estimator.ModeKeys.TRAIN)
        # Calculate loss.
        loss = model.loss(labels, 
                          resnet_feature_map, resnet_global_feaures)
        #获取训练步数
        global_step = tf.contrib.framework.get_or_create_global_step()
        # 一个mini-batch进行的训练
        train_op = one_step_train(loss, global_step)
        #返回所有summary对象先merge再serialize后的的字符串类型tensor
        summary_op = tf.summary.merge_all()
       
        
        
        #log_device_placement参数可以记录每一个操作使用的设备，这里的操作比较多，就不需要记录了，故设置为False  
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            #从头训练
            if train_from_start(FLAGS.ckpt_dir, MAX_STEP):
                print("start train from beginning")
                #☆☆☆☆☆☆☆☆saver的构建必须在所有的计算图都加载完之后才行,否则恢复计算图将没有被写入到ckpt中☆☆☆☆☆☆☆
                saver = tf.train.Saver()
                #变量初始化需要在所有op构建完成之后进行
                sess.run(tf.global_variables_initializer()) 
                
            #从存盘文件中加载进行训练
            else:
                # 获取全局步数
                ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir) # 通过检查点文件锁定最新的模型
                start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) + 1
                print("start train from step %d" % start_step)  
                
                '''
                #如果从存盘文件中加载计算图谱，则有问题：无法将images和labels加载进来
                meta_file = ckpt.model_checkpoint_path +'.meta'
                saver = tf.train.import_meta_graph(meta_file) 
                print("succsess restore the computation graph")
                '''
                variables_to_restore = tf.contrib.framework.get_variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess,ckpt.model_checkpoint_path) # 载入参数，参数保存在两个文件中，不过restore会自己寻找
                print("succsess restore the variable")
                global_step = tf.add(global_step, 1)
                print(global_step.eval())
                '''
                print('======trainable_variables=======')
                for v in tf.trainable_variables(): 
                    print(v.name)
               
                print('======model_variables=======')
                for v in tf.model_variables(): 
                    print(v.name)
                
                print('======global_variables=======')
                for v in tf.global_variables(): 
                    #print(v.name, v.eval())
                    print(v.name)
               
                print('=====moving_average_variables========')
                for v in tf.moving_average_variables():
                    print(v.name)
                
                print('======local_variables=======')
                for v in tf.local_variables():
                    print(v.name)
                '''
                
            #启动所有的queuerunners 
            tf.train.start_queue_runners(sess=sess) 
            
            train_writer = tf.summary.FileWriter(logdir=FLAGS.ckpt_dir + '/train', graph=sess.graph) 
            
            for step in range(start_step, MAX_STEP+1):  
                time_start = time.time()
                _, loss_value = sess.run(fetches=[train_op, loss]) 
                time_end = time.time()
                #用于验证当前迭代计算出的loss_value是否合理  
                assert not np.isnan(loss_value) 
                if step % 10 == 0:  
                    print('step %d, the loss_value is %.2f, costed time: %.3f ms/batch' % (step, loss_value, (time_end-time_start)*1000))  
                if step % 100 == 0:  
                    all_summaries = sess.run(summary_op)  
                    train_writer.add_summary(summary=all_summaries, global_step=step)  
                if step % 1000 == 0 or step == MAX_STEP:  
                    checkpoint_file = os.path.join(FLAGS.ckpt_dir, 'model.ckpt') #路径合并，返回合并后的字符串  
                    saver.save(sess, checkpoint_file, global_step=step)#把所有变量（包括moving average前后的模型参数）保存在variables_save_path路径下
            
            
            
def main(argv=None):
    train()
    


if __name__ == '__main__':
    tf.app.run()
