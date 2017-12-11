# -*- coding: utf-8 -*-
#===============================
#模型评估程序：
#    
#===============================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
import math

import numpy as np
import tensorflow as tf

import model
import train
import data_input

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


FLAGS = tf.app.flags.FLAGS
FLAGS.batch_size = model.FLAGS.batch_size
tf.app.flags.DEFINE_string('eval_dir', './CUHK03',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './CUHK03',
                           """Directory where to read model checkpoints.""")



def eval_once(saver, summary_writer, eval_op, summary_op):
    '''
    执行一次评估
    输入：
        saver：
        summary_writer：
        eval_op:最后一步评估的op
        summary_op：
    输出：
    '''
    with tf.Session() as sess:
        # 首先恢复模型
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        
        # 启动线程相关操作
        coord = tf.train.Coordinator()
        try:
            threads = []#线程列表
            #挨个启动线程
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord = coord, daemon=True, start=True))
            
            #得到需要迭代的次数，向上取整,基数不是偏大吗？
            num_iter = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size))
            total_sample_count = num_iter * FLAGS.batch_size     
            true_count = 0
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([eval_op])
                true_count += np.sum(predictions)
                step += 1
            
            # 计算精度
            precision = true_count / total_sample_count
            print('=======%s: precision @ 1 = %.5f' % (datetime.now(), precision))

            #添加summary操作
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
    
        except Exception as e:
            coord.request_stop(e)
        
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)



def evaluate():
    '''
    评估程序
    '''
    with tf.Graph().as_default() as g:
        data_type = 'test'
        images, labels = data_input.val_inputs(data_type, FLAGS.eval_dir, FLAGS.batch_size)
        
        logits = model.inference(images)
        
        #得分最高的下标值与labels是否相同
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        # tf.train.ExponentialMovingAverage类的一个用途就是可以用来恢复变量
        variable_averages = tf.train.ExponentialMovingAverage(
                train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir + '/eval', g)

        # 进行评估
        eval_once(saver, summary_writer, top_k_op, summary_op)
























def main(argv=None):  # pylint: disable=unused-argument
    evaluate()



if __name__ == '__main__':
  tf.app.run()