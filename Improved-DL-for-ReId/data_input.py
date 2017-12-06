# -*- coding: utf-8 -*-
#===============================
#数据读取程序：
#    从二进制文件中将数据读取出来，并做相应的数据增强处理
#===============================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from fnmatch import fnmatch

#裁剪输入给网络的尺寸
IMAGE_WITH = 60
IMAGE_HEIGHT = 160

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 98727#训练集样本数
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0#验证集集样本数
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 1440#测试集集样本数



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
    label_bytes = 1 #表明是negtive还是positive
    result.height = 176
    result.width = 66
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes * 2 #一对图像

    #定义一个reader,每次从文件中读取固定字节数
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  
    #返回从filename_queue中读取的(key, value)，key和value都是字符串类型的tensor,并且当队列中的某一个文件读完成时，该文件名会dequeue
    result.key, value = reader.read(filename_queue)

    #解码操作，看作读二进制文件，把字符串的字节转换成数值向量，每一个数值占用一个字节，在[0,255]区间内
    record_bytes = tf.decode_raw(value, tf.uint8)

    #从一维tensor对象中截取一个slice,将label和两个图片分别截取出来
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)#第二个参数表示截取的开始位置，第三个参数表示截取结束位置
    major1 =  tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes])#第二个参数表示截取的开始位置，第三个参数表示截取结束位置
    major2 =  tf.strided_slice(record_bytes, [label_bytes + image_bytes], [label_bytes + image_bytes + image_bytes])#第二个参数表示截取的开始位置，第三个参数表示截取结束位置

    #将数据进行变形为矩阵形式
    depth_major1 = tf.reshape(major1, [result.depth, result.height, result.width], name='major1_reshape')
    depth_major2 = tf.reshape(major2, [result.depth, result.height, result.width], name='major2_reshape')
  
    #对data维度进行重排序[depth, height, width] ==> [height, width, depth].
    result.uint8image1 = tf.transpose(depth_major1, [1, 2, 0])
    result.uint8image2 = tf.transpose(depth_major2, [1, 2, 0])


    return result



def _generate_image_and_label_batch(images, labels, min_queue_examples,
                                    batch_size, shuffle, use_fp16):
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
      use_fp16: 是否使用float16
    Returns:
      images: Images. 5D tensor of [batch_size, height, width, 3 x 2] size.
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
    #tf.train.batch()与tf.train.shuffle_batch()类似，之不多顺序地出队列（也即每次只能从一个data文件中读取batch）,少了随机性。
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
    images1, images2 = tf.split(images, num_or_size_splits = 2, axis = 3)#切分出来,master版本参数顺序改变了
    tf.summary.image('images1', images1)#输出预处理后图像的sumary缓存对象，用于在session中写入到事件文件中
    tf.summary.image('images2', images2)#输出预处理后图像的sumary缓存对象，用于在session中写入到事件文件中
    
    labels = tf.reshape(labels, [batch_size])

    if use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
        
    return images, labels



def train_inputs(file_prefix, data_dir, batch_size, use_fp16=False):
    """
    通过Reader ops为模型的创建构建训练数据

    Args:
        file_prefix:输入文件名的前缀，'train_set_1.bin'前缀为'train_set'后缀为'_1.bin'
        data_dir: Path to the data directory.
        batch_size: Number of images per batch.
        use_fp16: 是否使用float16
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WITH, 6] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    files = [files for files in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, files))]
    suffix = '_[0-9]*.bin'
    filenames = [f for f in files if fnmatch(f, file_prefix + suffix)]
    
    for f in filenames:
        if not tf.gfile.Exists(os.path.join(data_dir, f)):
            raise ValueError('Failed to find file: ' + os.path.join(data_dir, f))

    #将文件名输出到队列中，作为整个data pipe的第一阶段
    files = [os.path.join(data_dir, f) for f in filenames]#加上路径名
    filename_queue = tf.train.string_input_producer(files)

    #从文件名队列中读取一个tensor类型的图像
    read_input = read_data_set(filename_queue)
    
    reshaped_image1 = tf.cast(read_input.uint8image1, tf.float32)
    reshaped_image2 = tf.cast(read_input.uint8image2, tf.float32)

    height = IMAGE_HEIGHT
    width = IMAGE_WITH

    # 随机水平裁剪[height, width]的图片 
    distorted_image1 = tf.random_crop(reshaped_image1, [height, width, 3])
    distorted_image2 = tf.random_crop(reshaped_image2, [height, width, 3])

    # 随机左右翻转图像
    distorted_image1 = tf.image.random_flip_left_right(distorted_image1)
    distorted_image2 = tf.image.random_flip_left_right(distorted_image2)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    
    #设置随机亮度
    distorted_image1 = tf.image.random_brightness(distorted_image1, max_delta=63)
    distorted_image2 = tf.image.random_brightness(distorted_image2, max_delta=63)
    
    #设置随机对比度
    distorted_image1 = tf.image.random_contrast(distorted_image1, lower=0.2, upper=1.8)
    distorted_image2 = tf.image.random_contrast(distorted_image2, lower=0.2, upper=1.8)

    #使均值为0，方差为1，即对图像进行whiten操作，目的是降低输入图像的冗余性，尽量去除输入特征间的相关性
    float_image1 = tf.image.per_image_standardization(distorted_image1)
    float_image2 = tf.image.per_image_standardization(distorted_image2)

    # Set the shapes of tensors.
    float_image1.set_shape([height, width, 3])
    float_image2.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    #构建[IMAGE_HEIGHT, IMAGE_WITH, 6]形式的数据，合并两个图片
    images = tf.concat([float_image1, float_image2], 2)#合并两个图片,master版本中参数顺序变化了
    
    #用于确保读取到的batch中样例的随机性，使其覆盖到更多的类别、更多的数据文件！！！
    min_fraction_of_examples_in_queue = 0.1
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
    print ('Filling queue with %d images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(images, read_input.label,
                                 min_queue_examples, batch_size,
                                 False, use_fp16)



def val_inputs(file_prefix, data_dir, batch_size, use_fp16=False):
    """
    通过Reader ops为模型评估输入数据，
    构建过程中对图片进行了数据增强处理，处理过程没有训练时那么多

    Args:
      file_prefix:输入文件名的前缀，'train_set_1.bin'前缀为'train_set'后缀为'_1.bin'
      data_dir: Path to the  data directory.
      batch_size: Number of image-pair  per batch.
      use_fp16: 是否使用float16
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WITH, 6] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    files = [files for files in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, files))]
    suffix = '_[0-9]*.bin'
    filenames = [f for f in files if fnmatch(f, data_type + suffix)]

    if file_prefix.find('train'):
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    elif dfile_prefix.find('val'):
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    elif file_prefix.find('test'):
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TEST
    else:
        raise ValueError('not such data set:' + data_type )
        
    for f in filenames:
        if not tf.gfile.Exists(os.path.join(data_dir, f)):
            raise ValueError('Failed to find file: ' + os.path.join(data_dir, f))

    #将文件名输出到队列中，作为整个data pipe的第一阶段
    files = [os.path.join(data_dir, f) for f in filenames]#加上路径名
    filename_queue = tf.train.string_input_producer(files)
    
    # Read examples from files in the filename queue.
    read_input = read_data_set(filename_queue)
    
    reshaped_image1 = tf.cast(read_input.uint8image1, tf.float32)
    reshaped_image2 = tf.cast(read_input.uint8image2, tf.float32)

    height = IMAGE_HEIGHT
    width = IMAGE_WITH
    
    #裁取图像中间[height, width]部分
    resized_image1 = tf.image.resize_image_with_crop_or_pad(reshaped_image1, height, width)
    resized_image2 = tf.image.resize_image_with_crop_or_pad(reshaped_image2, height, width)

    #使均值为0，方差为1，即对图像进行whiten操作，目的是降低输入图像的冗余性，尽量去除输入特征间的相关性
    float_image1 = tf.image.per_image_standardization(resized_image1)
    float_image2 = tf.image.per_image_standardization(resized_image2)

    # Set the shapes of tensors.
    float_image1.set_shape([height, width, 3])
    float_image2.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    #构建[IMAGE_HEIGHT, IMAGE_WITH, 6]形式的数据，合并两个图片
    images = tf.concat([float_image1, float_image2], 2)#合并两个图片,master版本中参数顺序变化了

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(images, read_input.label,
                                 min_queue_examples, batch_size,
                                 False, use_fp16)
    


