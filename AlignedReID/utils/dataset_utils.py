#-*- coding:utf-8 -*-
#===================================
# utils for data set prepare
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
import os
import os.path as osp
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import threading
import Queue
import time
from collections import defaultdict
import shutil



def get_im_names(im_dir, pattern='*.jpg', return_np=True, return_path=False):
    """
    Get the image names in a dir. Optional to return numpy array, paths.
    """
    im_paths = glob.glob(osp.join(im_dir, pattern))
    im_names = [osp.basename(path) for path in im_paths]
    ret = im_paths if return_path else im_names
    if return_np:
        ret = np.array(ret)
    return ret


def parse_original_im_name(im_name, parse_type='id'):
    """
    Get the person id or cam from an image name.
    """
    assert parse_type in ('id', 'cam')
    if parse_type == 'id':
        parsed = -1 if im_name.startswith('-1') else int(im_name[:4])
    else:
        parsed = int(im_name[4]) if im_name.startswith('-1') else int(im_name[6])
    return parsed


def parse_full_path_im_name(im_name, parse_type='id'):
    """
    Get the person id or cam from an full path image name.
    """
    return parse_original_im_name(osp.basename(im_name), parse_type)


def partition_train_val_set(im_names, parse_im_name,
                            val_ids_num=None, val_prop=None, seed=1):
    """
    Partition the trainval set into train and val set. 
    Args:
        im_names: trainval image names
        parse_im_name: a function to parse id and camera from image name
        val_ids_num: number of ids for val set. If not set, val_prob is used.
        val_prop: the proportion of validation ids
        seed: the random seed to reproduce the partition results. If not to use, 
        then set to `None`.
    Returns:
        a dict with keys (`train_im_names`, 
                        `val_query_im_names`, 
                        `val_gallery_im_names`)
    """
    np.random.seed(seed)
    # Transform to numpy array for slicing.
    if not isinstance(im_names, np.ndarray):
        im_names = np.array(im_names)
    np.random.shuffle(im_names)
    ids = np.array([parse_im_name(n, 'id') for n in im_names])
    cams = np.array([parse_im_name(n, 'cam') for n in im_names])
    unique_ids = np.unique(ids)
    np.random.shuffle(unique_ids)

    # Query indices and gallery indices
    query_inds = []
    gallery_inds = []

    if val_ids_num is None:
        assert 0 < val_prop < 1
        val_ids_num = int(len(unique_ids) * val_prop)
    num_selected_ids = 0
    for unique_id in unique_ids:
        query_inds_ = []
        # The indices of this id in trainval set.
        inds = np.argwhere(unique_id == ids).flatten()
        # The cams that this id has.
        unique_cams = np.unique(cams[inds])
        # For each cam, select one image for query set.
        for unique_cam in unique_cams:
            query_inds_.append(
                inds[np.argwhere(cams[inds] == unique_cam).flatten()[0]])
        gallery_inds_ = list(set(inds) - set(query_inds_))
        # For each query image, if there is no same-id different-cam images in
        # gallery, put it in gallery.
        for query_ind in query_inds_:
            if len(gallery_inds_) == 0 \
                or len(np.argwhere(cams[gallery_inds_] != cams[query_ind])
                            .flatten()) == 0:
                query_inds_.remove(query_ind)
                gallery_inds_.append(query_ind)
        # If no query image is left, leave this id in train set.
        if len(query_inds_) == 0:
            continue
        query_inds.append(query_inds_)
        gallery_inds.append(gallery_inds_)
        num_selected_ids += 1
        if num_selected_ids >= val_ids_num:
            break

    query_inds = np.hstack(query_inds)
    gallery_inds = np.hstack(gallery_inds)
    val_inds = np.hstack([query_inds, gallery_inds])
    trainval_inds = np.arange(len(im_names))
    train_inds = np.setdiff1d(trainval_inds, val_inds)

    train_inds = np.sort(train_inds)
    query_inds = np.sort(query_inds)
    gallery_inds = np.sort(gallery_inds)

    partitions = dict(train_im_names=im_names[train_inds],
                        val_query_im_names=im_names[query_inds],
                        val_gallery_im_names=im_names[gallery_inds])

    return partitions