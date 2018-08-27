#-*- coding:utf-8 -*-
#===================================
# common utils for other programs
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
import sys
import os
import os.path as osp
if sys.version_info.major == 2:
    import pickle
else:
    import cPickle as pickle
import numpy as np
from scipy import io
import datetime
import time
from contextlib import contextmanager
import cv2

import torch
from torch.autograd import Variable


def time_str(fmt=None):
  if fmt is None:
    fmt = '%Y-%m-%d_%H:%M:%S'
  return datetime.datetime.today().strftime(fmt)


def tight_float_str(x, fmt='{:.4f}'):
  return fmt.format(x).rstrip('0').rstrip('.')


def load_pickle(path):
    """
    Check and load pickle object.
    According to this post: https://stackoverflow.com/a/41733927, cPickle and 
    disabling garbage collector helps with loading speed."""
    assert osp.exists(path)
    # gc.disable()
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    # gc.enable()
    return ret


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


# Great idea from https://github.com/amdegroot/ssd.pytorch
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Logger(object):
    """Modified from Tong Xiao's `Logger` in open-reid.
    This class overwrites sys.stdout or sys.stderr, so that console logs can
    also be written to file.
    Args:
        fpath: file path
        console: one of ['stdout', 'stderr']
        immediately_visible: If `False`, the file is opened only once and closed
        after exiting. In this case, the message written to file may not be
        immediately visible (Because the file handle is occupied by the
        program?). If `True`, each writing operation of the console will
        open, write to, and close the file. If your program has tons of writing
        operations, the cost of opening and closing file may be obvious. (?)
    Usage example:
        Logger('stdout.txt', 'stdout', False)
        Logger('stderr.txt', 'stderr', False)
    NOTE: File will be deleted if already existing. Log dir and file is created
        lazily -- if no message is written, the dir and file will not be created.
    """

    def __init__(self, fpath=None, console='stdout', immediately_visible=False):
        import sys
        import os
        import os.path as osp

        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == 'stdout' else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visible = immediately_visible
        if fpath is not None:
            # Remove existing log file.
            if osp.exists(fpath):
                os.remove(fpath)

        # Overwrite
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            may_make_dir(os.path.dirname(osp.abspath(self.file)))
        if self.immediately_visible:
            with open(self.file, 'a') as f:
                f.write(msg)
        else:
            if self.f is None:
                self.f = open(self.file, 'w')
            self.f.write(msg)

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()


def set_devices(sys_device_ids):
    """
    It sets some GPUs to be visible and returns some wrappers to transferring 
    Variables/Tensors and Modules/Optimizers.
    Args:
        sys_device_ids: a tuple; which GPUs to use
        e.g.  sys_device_ids = (), only use cpu
                sys_device_ids = (3,), use the 4th gpu
                sys_device_ids = (0, 1, 2, 3,), use first 4 gpus
                sys_device_ids = (0, 2, 4,), use the 1st, 3rd and 5th gpus
    """
    # Set the CUDA_VISIBLE_DEVICES environment variable
    import os
    visible_devices = ''
    for i in sys_device_ids:
        visible_devices += '{}, '.format(i)
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices



def set_seed(seed):
    '''
    set seed for moudel random, numpy, torch(cup and gpus)
    '''
    import random
    random.seed(seed)
    print('setting random-seed to {}'.format(seed))

    import numpy as np
    np.random.seed(seed)
    print('setting np-random-seed to {}'.format(seed))

    # set seed for CPU
    torch.manual_seed(seed)
    print('setting torch-seed to {}'.format(seed))
    try:
        # set seed for all visible GPUs
        torch.cuda.manual_seed_all(seed)
        print('setting torch-cuda-seed to {}'.format(seed))
    except:
        pass


def find_index(seq, item):
    for i, x in enumerate(seq):
        if item == x:
            return i
    return -1


def to_scalar(vt):
    """
    Transform a length-1 pytorch Variable or Tensor to scalar. 
    Suppose tx is a torch Tensor with shape tx.size() = torch.Size([1]), 
    then npx = tx.cpu().numpy() has shape (1,), not 1.
    """
    if isinstance(vt, Variable):
        return vt.data.cpu().numpy().flatten()[0]
    if torch.is_tensor(vt):
        return vt.cpu().numpy().flatten()[0]
    raise TypeError('Input should be a variable or tensor')


@contextmanager
def measure_time(enter_msg):
    st = time.time()
    print(enter_msg)
    yield
    print('Done, {:.3}s'.format(time.time() - st))



def pre_process_im(im_path,
                resize_size,
                im_mean=[0.485, 0.456, 0.406],
                im_std=[0.229, 0.224, 0.225],
                batch_dims = 'NCHW'):
    """
    Pre-process image.
    args:
        im_path: image path or im array
        resize_size: (height, width)
        im_mean: the mean of the image
        im_std: the std of the image
        batch_dims: indicate wether 'NCHW' or 'NHWC'
    """
    if type(im_path) == str:
        im = cv2.imread(im_path)
    else:
        im = im_path
    im = im[:,:,[2,1,0]]
    # Resize.
    if resize_size is not None:
        im = cv2.resize(im, (resize_size[1],resize_size[0] ), interpolation=cv2.INTER_LINEAR)

    # Subtract mean and scaled by std
    # im -= np.array(self.im_mean) # This causes an error:
    # Cannot cast ufunc subtract output from dtype('float64') to
    # dtype('uint8') with casting rule 'same_kind'
    if im_mean is not None:
        # scaled by 1/255.
        im = im / 255.
        im = im - np.array(im_mean)
    if im_mean is not None and im_std is not None:
        im = im / np.array(im_std).astype(float)

    # The original image has dims 'HWC', transform it to 'CHW'.
    if batch_dims == 'NCHW':
        im = im.transpose(2, 0, 1)

    return im