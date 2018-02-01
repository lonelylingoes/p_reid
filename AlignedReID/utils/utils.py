#-*- coding:utf-8 -*-
#===================================
# utils for data set prepare
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
import sys
import os
import os.path as osp
import cPickle as pickle
import gc
import numpy as np
from scipy import io
import datetime
import time
from contextlib import contextmanager

import torch
from torch.autograd import Variable


def may_make_dir(path):
    """
    Args:
        path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
    Note:
        `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
    """

    assert path not in [None, '']

    if not osp.exists(path):
        os.makedirs(path)


def save_pickle(obj, path):
    """
    Create dir and save file.
    """
    may_make_dir(osp.dirname(osp.abspath(path)))
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)