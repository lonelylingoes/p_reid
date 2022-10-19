#-*- coding:utf-8 -*-
#===================================
#msmt17 data set prepare program
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

import tarfile
import numpy as np

from reid_utils.common_utils import may_make_dir
from reid_utils.common_utils import save_pickle


from reid_utils.dataset_utils import partition_train_val_set
from reid_utils.dataset_utils import parse_original_msmt17_im_name
from reid_utils.dataset_utils import parse_full_path_msmt17_im_name


def get_im_names(im_dir, list_file, return_np=True, return_path=True):
    """
    Get the image names in a dir. Optional to return numpy array, paths.
    """
    path_names = []
    file_names = []
    with open(list_file, 'r') as f:
        for line in f:
            file_name, _ = line.split(' ')
            path_name = osp.join(im_dir, file_name)
            path_names.append(path_name)
            file_names.append(osp.basename(path_name))
    ret = path_names if return_path else file_names
    if return_np:
        ret = np.array(ret)
    return ret


def get_images_split(zip_file, save_dir=None):
    """
    Rename and move all used images to a directory.
    """

    print("Extracting zip file")
    root = osp.dirname(osp.abspath(zip_file))
    if save_dir is None:
        save_dir = root
    may_make_dir(osp.abspath(save_dir))
    with tarfile.open(zip_file) as z:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(z, path=save_dir)
    print("Extracting zip file done")

    raw_dir = osp.join(save_dir, osp.basename(zip_file)[:-7])

    im_paths = []
    nums = []

    # train set
    im_paths_ = get_im_names(osp.join(raw_dir, 'train'), osp.join(raw_dir, 'list_train.txt'),
                            return_path=True, return_np=False)
    im_paths_.sort()
    im_paths += list(im_paths_)
    nums.append(len(im_paths_))

    # val set
    im_paths_ = get_im_names(osp.join(raw_dir, 'train'), osp.join(raw_dir, 'list_val.txt'),
                            return_path=True, return_np=False)
    im_paths_.sort()
    im_paths += list(im_paths_)
    nums.append(len(im_paths_))

    # query set 
    im_paths_ = get_im_names(osp.join(raw_dir, 'test'), osp.join(raw_dir, 'list_query.txt'),
                            return_path=True, return_np=False)
    im_paths_.sort()
    im_paths += list(im_paths_)
    nums.append(len(im_paths_))

    # gallery set 
    im_paths_ = get_im_names(osp.join(raw_dir, 'test'), osp.join(raw_dir, 'list_gallery.txt'),
                            return_path=True, return_np=False)
    im_paths_.sort()
    im_paths += list(im_paths_)
    nums.append(len(im_paths_))

    split = dict()
    keys = ['train_im_names', 'val_im_names', 'gallery_im_names', 'q_im_names']
    inds = [0] + nums
    inds = np.cumsum(np.array(inds))# get index position
    for i, k in enumerate(keys):
        split[k] = im_paths[inds[i]:inds[i + 1]]

    split['trainval_im_names'] = split['train_im_names'] + split['val_im_names']

    return split


def transform(zip_file, save_dir=None):
    """
    Refactor file directories, partition the train/val/test set.
    """
    train_test_split = get_images_split(zip_file, save_dir)
  
    # == partition train/val/ set ==
    # get the trainval_ids by set data structure
    trainval_ids = list(set([parse_full_path_msmt17_im_name(n, 'id')
                            for n in train_test_split['trainval_im_names']]))
    # Sort ids, so that id-to-label mapping remains the same when running
    # the code on different machines.
    trainval_ids.sort()
    # trans the ids to lables
    trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))

    partitions = partition_train_val_set(
                    train_test_split['val_im_names'], parse_full_path_msmt17_im_name, val_prop=1)

    train_ids = list(set([parse_full_path_msmt17_im_name(n, 'id')
                        for n in train_test_split['train_im_names']]))
    # Sort ids, so that id-to-label mapping remains the same when running
    # the code on different machines.
    train_ids.sort()
    train_ids2labels = dict(zip(train_ids, range(len(train_ids))))

    # A mark is used to denote whether the image is from
    #   query (mark == 0), or
    #   gallery (mark == 1), or
    #   multi query (mark == 2) set

    val_marks = [0, ] * len(partitions['val_query_im_names']) \
                + [1, ] * len(partitions['val_gallery_im_names'])
    val_im_names = list(partitions['val_query_im_names']) \
                    + list(partitions['val_gallery_im_names'])

    test_im_names = list(train_test_split['q_im_names']) \
                    + list(train_test_split['gallery_im_names'])
    test_marks = [0, ] * len(train_test_split['q_im_names']) \
                + [1, ] * len(train_test_split['gallery_im_names'])

    partitions = {'trainval_im_names': train_test_split['trainval_im_names'],
                    'trainval_ids2labels': trainval_ids2labels,
                    'train_im_names': train_test_split['train_im_names'],
                    'train_ids2labels': train_ids2labels,
                    'val_im_names': val_im_names,
                    'val_marks': val_marks,
                    'test_im_names': test_im_names,
                    'test_marks': test_marks}
    partition_file = osp.join(save_dir, 'partitions.pkl')
    save_pickle(partitions, partition_file)
    print('Partition file saved to {}'.format(partition_file))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transform MSMT17 Dataset")
    parser.add_argument('--zip_file', type=str,
                        default='/data/DataSet/msmt17/MSMT17_V1.tar.gz')
    parser.add_argument('--save_dir', type=str,
                        default='/data/DataSet/msmt17')
    args = parser.parse_args()
    zip_file = osp.abspath(osp.expanduser(args.zip_file))
    save_dir = osp.abspath(osp.expanduser(args.save_dir))
    transform(zip_file, save_dir)
