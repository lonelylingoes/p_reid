# -*- coding: utf-8 -*-
#===============================
#模型多GPU训练程序
#===============================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
import re
import data_input
import model
import train


FLAGS = tf.app.flags.FLAGS
FLAGS.batch_size = model.FLAGS.batch_size
FLAGS.use_fp16 = model.FLAGS.use_fp16

tf.app.flags.DEFINE_string('ckpt_dir', './CUHK03',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('input_dir', './CUHK03',
                           """Directory where to input data.""")
tf.app.flags.DEFINE_integer('max_steps', 210000,
                            """Number of batches to run.""")#根据mini-batch为128的比例算了一下为252000
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            """How many GPUs to use.""")

IMAGE_WITH = data_input.IMAGE_WITH
IMAGE_HEIGHT = data_input.IMAGE_HEIGHT

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
LEARNING_RATE_DECAY_RATE = 0.0001  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.
EXPONENT_RATE = 0.75                #论文中的指数项


def _inverse_time_exponent_decay(learning_rate, global_step, decay_rate, exponent_rate,
                                 name=None):
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
    with ops.name_scope(name, "InverseTimeDecay",
              [learning_rate, global_step, decay_rate, exponent_rate]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        decay_rate = math_ops.cast(decay_rate, dtype)
        exponent_rate = math_ops.cast(exponent_rate, dtype)
        const = math_ops.cast(constant_op.constant(1), learning_rate.dtype)
        denom = math_ops.add(const, math_ops.multiply(decay_rate, global_step))
        exponent = math_ops.pow(denom, exponent_rate)
        return math_ops.div(learning_rate, exponent, name=name)


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



def tower_loss(scope, images, labels):
    '''
    Calculate the total loss on a single tower running the CIFAR model.

    Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        images: Images. 4D tensor of shape [batch_size, height, width, 6].
        labels: Labels. 1D tensor of shape [batch_size].

     Returns:
        Tensor of shape [] containing the total loss for a batch of data
    '''

    # Build inference Graph.
    logits = model.inference(images)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    # 调用该函数并没有使用最终的计算结果，仅仅是利用它构建的计算loss的计算图
    _ = model.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    # 只取当前tower的loss,为了区分不同GPU
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

     return total_loss



def average_gradients(tower_grads):
    '''
    Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.
    该函数采用同步的方式计算梯度

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    '''
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        #   共有N个这样的数据，其中N是变量的数目
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        # 由于变量是共享的，所以只取第一个元素就够了
        v = grad_and_vars[0][1]# 第一个下标取得是(grad0_gpu0, var0_gpu0),第二个下标取的是该元祖中的变量
        grad_and_var = (grad, v)# 还要组织成该元祖形式
        average_grads.append(grad_and_var)

    return average_grads

   
    


def train():
    '''
    多GPU训练
    '''
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.

        # 获取训练步数
        global_step = tf.contrib.framework.get_or_create_global_step()
        #计算衰减过后的学习率
        lr = _inverse_time_exponent_decay(INITIAL_LEARNING_RATE, global_step, LEARNING_RATE_DECAY_RATE, EXPONENT_RATE)
        tf.summary.scalar('learning_rate', lr)
        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)

        # Get images and labels
        images, labels = data_input.train_inputs(FLAGS.input_dir, FLAGS.batch_size)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
              [images, labels], capacity=2 * FLAGS.num_gpus)
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
                        # Dequeues one batch for the GPU
                        image_batch, label_batch = batch_queue.dequeue()
                        # Calculate the loss for one tower of the model. This function
                        # constructs the entire model but shares the variables across
                        # all towers.
                        # 该函数构建了整个模型
                        loss = tower_loss(scope, image_batch, label_batch)

                        # Reuse variables for the next tower.
                        # 允许变量共享
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        # collection中的scope信息是怎么添加进去的？
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this tower.
                        grads_and_vars = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads_and_vars)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            model.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        #训练步数起始值
        start_step = 1

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement)) as sess:
            # 从头训练
            if train_from_start(FLAGS.input_dir, FLAGS.max_steps):
                print("========start train from beginning")
                #☆☆☆☆☆☆☆☆saver的构建必须在所有的计算图都加载完之后才行,否则恢复计算图将没有被写入到ckpt中☆☆☆☆☆☆☆
                # Create a saver.
                saver = tf.train.Saver()

                # Build an initialization operation to run below.
                init = tf.global_variables_initializer()

                sess.run(init)
            else:
                # 获取全局步数
                ckpt = tf.train.get_checkpoint_state(FLAGS.input_dir) # 通过检查点文件锁定最新的模型
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
                print("=======succsess restore the variable")
                global_step = tf.add(global_step, 1)
                

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.ckpt_dir, sess.graph)

        for step in range(start_step, FLAGS.max_steps+1, FLAGS.num_gpus):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if (step+FLAGS.num_gpus-1) % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (step+FLAGS.num_gpus-1, loss_value,
                                     examples_per_sec, sec_per_batch))

            if (step+FLAGS.num_gpus-1) % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step+FLAGS.num_gpus-1)

            # Save the model checkpoint periodically.
            if (step+FLAGS.num_gpus-1) % 1000 == 0 or (step+FLAGS.num_gpus-1 + FLAGS.num_gpus) >= FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step+FLAGS.num_gpus-1)


            
            

def main(argv=None):
    train()
    


if __name__ == '__main__':
    tf.app.run()
