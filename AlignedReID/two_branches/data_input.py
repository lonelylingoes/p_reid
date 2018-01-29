# -*- coding: utf-8 -*-
#===============================
#数据读取程序：
#    从二进制文件中将数据读取出来
#===============================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append('../')
import os
import tensorflow as tf

from utils import read_conf

#输入的图片尺寸
IMAGE_WITH = read_conf.get_conf_int_param('cnn_param', 'input_with')
IMAGE_HEIGHT = read_conf.get_conf_int_param('cnn_param', 'input_height')
DATA_FORMAT = read_conf.get_conf_str_param('cnn_param', 'data_format')
IMAGES_PER_ID = read_conf.get_conf_int_param('hyper_param', 'images_per_id')
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = read_conf.get_conf_int_param('hyper_param', 'pictures_per_epoch')


def read_data_set(filename_queue):
    """
    从文件队列中读取数据
  
    Args:
    filename_queue: A queue of strings with the filenames to read from.
    
      Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number for this example.
      uint8image1: a [height, width, depth] uint8 Tensor with the image data1
      uint8image2: a [height, width, depth] uint8 Tensor with the image data2
      label: an int32 Tensor with the label in the 1 or 0.
    """
    #定义一个空类
    class PAIRRecord(object):
        pass
    result = PAIRRecord()
  
    #图片的尺寸，这取决于之前数据预处理的
    label_bytes = 4 #表明是negtive还是positive
    result.height = IMAGE_HEIGHT 
    result.width = IMAGE_WITH
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = (label_bytes + image_bytes) #一张图像

    #定义一个reader,每次从文件中读取固定字节数
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  
    #返回从filename_queue中读取的(key, value)，key和value都是字符串类型的tensor,并且当队列中的某一个文件读完成时，该文件名会dequeue
    result.key, value = reader.read(filename_queue)

    #解码操作，看作读二进制文件，把字符串的字节转换成数值向量，每一个数值占用一个字节，在[0,255]区间内
    record_bytes = tf.decode_raw(value, tf.uint8)

    #从一维tensor对象中截取一个slice,将label和图片分别截取出来
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)#第二个参数表示截取的开始位置，第三个参数表示截取结束位置
    depth_major = tf.reshape(
            tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])    
    
    #对data维度进行重排序[depth, height, width] ==> [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
# =============================================================================
#     if DATA_FORMAT != 'channels_first':
#         #对data维度进行重排序[depth, height, width] ==> [height, width, depth].
#         result.uint8image = tf.transpose(depth_major, [1, 2, 0])
#     else:
#         result.uint8image = depth_major
# =============================================================================
  
    return result



def _generate_image_and_label_batch(images, labels, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images pair.
       疑问：
           线程数与reader个数的关系？一个线程就是一个reader?
    Args:
      images: 4-D Tensors of [2, height, width, 3] of type.float32.
      labels: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3 ] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' image_pairs from the example queue.
    num_preprocess_threads = 8
    
    #tf.train.shuffle_batch()函数用于随机地shuffling队列中的tensor来创建batch(也即每次可以读取多个data文件中的样例构成一个batch)
    #使用该函数，避免了两个不同的线程从同一个文件中读取同一个样本,也即一个文件同时只有一个线程在读
	 #这个函数向当前Graph中添加了下列对象：
	 #  *创建了一个shuffling queue,用于把'tensors'中的tensors压入该队列；
	 #  *一个dequeue_many操作，用于根据队列中的数据创建一个batch；
	 #  *创建了一个QueueRunner对象，用于启动一个进程压数据到队列
	 #capacity:参数用于控制shuffling queue的最大长度；
	 #min_after_dequeue:参数表示进行一次dequeue操作后队列中元素的最小数量，可以用于确保batch中元素的随机性
	 #num_threads:参数用于制定多少个threads负责压tensors到队列；
	 #enqueue_many:参数用于表征是否tensors中的每一个tensor都代表一个样例
    #-------------------------
    #tf.train.batch()与tf.train.shuffle_batch()类似，只不过tf.train.batch()顺序地出队列（也即每次只能从一个data文件中读取batch）,少了随机性。
    if shuffle:
        images, labels = tf.train.shuffle_batch(
                [images, labels],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
    else:
        images, labels = tf.train.batch(
                [images, labels],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)#输出预处理后图像的sumary缓存对象，用于在session中写入到事件文件中
    tf.summary.scalar('labels', labels)#输出预处理后label的sumary缓存对象，用于在session中写入到事件文件中
    
    labels = tf.reshape(labels, [batch_size])

    return images, labels



def train_inputs(data_dir_list, batch_size):
    """
    通过Reader ops为模型的创建构建训练数据

    Args:
        data_dir_list: 数据路径集合.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WITH, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = []
    for path in data_dir_list:#遍历每个数据集
        for file_info in os.walk(path):
            for file in file_info[2]:
                filenames.append(os.path.join(file_info[0], file))
    
    #从文件名队列中读取一个tensor类型的图像
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_data_set(filename_queue)
    
    float_image = tf.cast(read_input.uint8image, tf.float32)
     
    height = IMAGE_HEIGHT
    width = IMAGE_WITH

    # Set the shapes of tensors.
# =============================================================================
#     if DATA_FORMAT != 'channels_first':
#         float_image.set_shape([height, width, 3])
#     else:
#         float_image.set_shape([3, height, width])
# =============================================================================
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    #用于确保读取到的batch中样例的随机性，使其覆盖到更多的类别、更多的数据文件！！！
    min_fraction_of_examples_in_queue = 0.1
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
    print ('Filling queue with %d images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                 min_queue_examples, batch_size,
                                 False)

