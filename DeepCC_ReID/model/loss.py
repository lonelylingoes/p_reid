#-*- coding:utf-8 -*-
#===================================
# loss program
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import sys
sys.path.append('../')



class TripletLoss(object):
    """
    Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
    Loss for Person Re-Identification'.
    """
    def __init__(self, margin=1):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=1)

    def __call__(self, dist_ap, dist_an):
        """
        Args:
        dist_ap: pytorch Variable, distance between anchor and positive sample, 
            shape [N]
        dist_an: pytorch Variable, distance between anchor and negative sample, 
            shape [N]
        Returns:
        loss: pytorch Variable, with shape [1]
        """
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss


def normalize(x, axis=-1):
    """
    Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch Variable
    Returns:
        x: pytorch Variable, same shape as input      
    """
    x = x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def normalize_np(nparray, order=2, axis=0):
    """
    Normalize a N-D numpy array along the specified axis.
    """
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(x, y, type='euclidean'):
    """
    avoid loop, promote the speed
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
        type: one of ['cosine', 'euclidean']
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    assert type in ['cosine', 'euclidean']
    if type == 'cosione':
        pass
    else:
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1).expand(m, n)
        yy = torch.pow(y, 2).sum(1).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist


def compute_dist_np(array1, array2, type='euclidean'):
    """
    Compute the euclidean or cosine distance of all pairs.
    Args:
        array1: numpy array with shape [m1, n]
        array2: numpy array with shape [m2, n]
        type: one of ['cosine', 'euclidean']
    Returns:
        numpy array with shape [m1, m2]
    """
    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        array1 = normalize_np(array1, axis=1)
        array2 = normalize_np(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist


def batch_compute_dist(x, y, type='euclidean'):
    """
    Args:
        x: pytorch Variable, with shape [N, m, d]
        y: pytorch Variable, with shape [N, n, d]
        type: one of ['cosine', 'euclidean']
    Returns:
        dist: pytorch Variable, with shape [N, m, n]
    """
    assert type in ['cosine', 'euclidean']
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    N, m, d = x.size()
    N, n, d = y.size()

    if type == 'cosine':
        pass
    else:
        # shape [N, m, n]
        xx = torch.pow(x, 2).sum(-1,keepdim=True).expand(N, m, n)
        yy = torch.pow(y, 2).sum(-1,keepdim=True).expand(N, n, m).permute(0, 2, 1)
        dist = xx + yy
        dist.baddbmm_(1, -2, x, y.permute(0, 2, 1))
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist


def shortest_dist(dist_mat):
    """
    Parallel version. arroding to the paper
    Args:
        dist_mat: pytorch Variable, available shape:
        1) [m, n]
        2) [m, n, N], N is batch size
        3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
        dist: three cases corresponding to `dist_mat`:
        1) scalar
        2) pytorch Variable, with shape [N]
        3) pytorch Variable, with shape [*]
    """
    m, n = dist_mat.size()[:2]
    # Just offering some reference for accessing intermediate distance.
    dist = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i][j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
            else:
                dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
    dist = dist[-1][-1]
    return dist



def shortest_dist_np(dist_mat):
    """
    Parallel version.
    Args:
        dist_mat: numpy array, available shape
        1) [m, n]
        2) [m, n, N], N is batch size
        3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
        dist: three cases corresponding to `dist_mat`
        1) scalar
        2) numpy array, with shape [N]
        3) numpy array with shape [*]
    """
    m, n = dist_mat.shape[:2]
    dist = np.zeros_like(dist_mat)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i, j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
            else:
                dist[i, j] = \
                    np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
                    + dist_mat[i, j]
    # I ran into memory disaster when returning this reference! I still don't
    # know why.
    # dist = dist[-1, -1]
    dist = dist[-1, -1].copy()
    return dist


def local_dist_mat(x, y):
    """
    generate local distance matrix for mining hard exaple.
    normally, use batch_local_dist
    Args:
        x: pytorch Variable, with shape [M, m, d]
        y: pytorch Variable, with shape [N, n, d]
    Returns:
        dist: pytorch Variable, with shape [M, N]
    """
    M, m, d = x.size()
    N, n, d = y.size()
    x = x.contiguous().view(M * m, d)
    y = y.contiguous().view(N * n, d)
    # shape [M * m, N * n]
    dist_mat = compute_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
    dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
    # shape [M, N]
    dist_mat = shortest_dist(dist_mat)
    return dist_mat


def meta_local_dist_np(x, y):
    """
    Args:
        x: numpy array, with shape [m, d]
        y: numpy array, with shape [n, d]
    Returns:
        dist: scalar
    """
    eu_dist = compute_dist_np(x, y, 'euclidean')
    dist_mat = (np.exp(eu_dist) - 1.) / (np.exp(eu_dist) + 1.)
    dist = shortest_dist_np(dist_mat[np.newaxis])[0]
    return dist


def local_dist_mat_np(x, y):
    """
    Parallel version.
    generate local distance matrix for mining hard exaple.
    Args:
        x: numpy array, with shape [M, m, d]
        y: numpy array, with shape [N, n, d]
    Returns:
        dist: numpy array, with shape [M, N]
    """
    M, m, d = x.shape
    N, n, d = y.shape
    x = x.reshape([M * m, d])
    y = y.reshape([N * n, d])
    # shape [M * m, N * n]
    dist_mat = compute_dist_np(x, y, type='euclidean')
    dist_mat = (np.exp(dist_mat) - 1.) / (np.exp(dist_mat) + 1.)
    # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
    dist_mat = dist_mat.reshape([M, m, N, n]).transpose([1, 3, 0, 2])
    # shape [M, N]
    dist_mat = shortest_dist_np(dist_mat)
    return dist_mat


def local_dist_np(x, y):
    if (x.ndim == 2) and (y.ndim == 2):
        return meta_local_dist_np(x, y)
    elif (x.ndim == 3) and (y.ndim == 3):
        return local_dist_mat_np(x, y)
    else:
        raise NotImplementedError('Input shape not supported.')


def batch_local_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [N, m, d]
        y: pytorch Variable, with shape [N, n, d]
    Returns:
        dist: pytorch Variable, with shape [N]
    """
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    # shape [N, m, n]
    dist_mat = batch_compute_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    # shape [N]
    dist = shortest_dist(dist_mat.permute(1, 2, 0))
    return dist



def low_memory_matrix_op(
    func,
    x, y,
    x_split_axis, y_split_axis,
    x_num_splits, y_num_splits,
    verbose=False):
    """
    For matrix operation like multiplication, in order not to flood the memory 
    with huge data, split matrices into smaller parts (Divide and Conquer). 
    
    Note: 
        If still out of memory, increase `*_num_splits`.
    
    Args:
        func: a matrix function func(x, y) -> z with shape [M, N]
        x: numpy array, the dimension to split has length M
        y: numpy array, the dimension to split has length N
        x_split_axis: The axis to split x into parts
        y_split_axis: The axis to split y into parts
        x_num_splits: number of splits. 1 <= x_num_splits <= M
        y_num_splits: number of splits. 1 <= y_num_splits <= N
        verbose: whether to print the progress
        
    Returns:
        mat: numpy array, shape [M, N]
    """
    if verbose:
        import sys
        import time
        printed = False
        st = time.time()
        last_time = time.time()

    mat = [[] for _ in range(x_num_splits)]
    for i, part_x in enumerate(
        np.array_split(x, x_num_splits, axis=x_split_axis)):
        for j, part_y in enumerate(np.array_split(y, y_num_splits, axis=y_split_axis)):
            part_mat = func(part_x, part_y)
        mat[i].append(part_mat)

        if verbose:
            if not printed:
                printed = True
            else:
                # Clean the current line
                sys.stdout.write("\033[F\033[K")
            print('Matrix part ({}, {}) / ({}, {}), +{:.2f}s, total {:.2f}s'
                .format(i + 1, j + 1, x_num_splits, y_num_splits,
                        time.time() - last_time, time.time() - st))
            last_time = time.time()
        mat[i] = np.concatenate(mat[i], axis=1)
    mat = np.concatenate(mat, axis=0)
    return mat



def weighted_distance(dist_mat, labels):
    '''
    get the weighted distance.
    args:
        dist_mat: the distance mat, with shape[N, N]
        labels: pytorch LongTensor, with shape [N]
    '''
    N = dist_mat.size(0)
    # shape [N, K]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    # shape [N, N-K]
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # shape [N, K]
    w_p = F.softmax(dist_mat[is_pos].view(N, -1), dim = 1)
    # shape [N, N-K]
    W_n = F.softmax(dist_mat[is_neg].view(N, -1), dim = 1)

    # shape [N,]
    dist_ap = torch.sum(torch.dot(w_p, dist_mat[is_pos].view(N, -1)), dim = 1)
    dist_an = torch.sum(torch.dot(w_n, dist_mat[is_neg].view(N, -1)), dim = 1)

    return dist_ap, dist_an



def global_loss(tri_loss, global_feat, labels, normalize_feature=True):
    """
    Args:
        tri_loss: a `TripletLoss` object
        global_feat: pytorch Variable, shape [N, C]
        labels: pytorch LongTensor, with shape [N]
        normalize_feature: whether to normalize feature to unit length along the 
        Channel dimension
    Returns:
        loss: pytorch Variable, with shape [1]
        =============
        For Debugging
        =============
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        ===================
        For Mutual Learning
        ===================
        dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
    """
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    # beacause in class Model, global feature has been squeezed,
    # so here, call compute_dist(), otherwise, batch_compute_dist() should be called
    dist_mat = compute_dist(global_feat, global_feat)
    dist_ap, dist_an = weighted_distance(dist_mat, lables)
    loss = tri_loss(dist_ap, dist_an)

    return loss, dist_ap, dist_an, dist_mat


def local_loss(tri_loss, local_feat, labels, normalize_feature=True):
    """
    Args:
        tri_loss: a `TripletLoss` object
        local_feat: pytorch Variable, shape [N, H, c] (NOTE THE SHAPE!)
        labels: pytorch LongTensor, with shape [N]
        normalize_feature: whether to normalize feature to unit length along the 
        Channel dimension
    
    If hard samples are specified by `p_inds` and `n_inds`, then `labels` is not 
    used. Otherwise, local distance finds its own hard samples independent of 
    global distance.
    
    Returns:
        loss: pytorch Variable,with shape [1]
        ===================
        For Mutual Learning
        ===================
        dist_mat: pytorch Variable, pairwise local distance; shape [N, N]
    """
    if normalize_feature:
        local_feat = normalize(local_feat, axis=-1)
    
    dist_mat = local_dist_mat(local_feat, local_feat)
    dist_ap, dist_an = weighted_distance(dist_mat, lables)
    loss = tri_loss(dist_ap, dist_an)
    return loss, dist_ap, dist_an, dist_mat


def total_loss(loss_dict, global_feat, local_feat, logits, labels, cfg):
    '''
    compute total loss by the cfg
    args:
        loss_dict: a dict contains the 'loss' object
        global_feat: global feature
        local_feat: local feature
        logits: classfication logits
        labels: identy labels, [variable]
        cfg:    config
    '''

    g_loss, p_inds, n_inds, g_dist_ap, g_dist_an, g_dist_mat = global_loss(
        loss_dict['g_tri_loss'], global_feat, labels,
        normalize_feature=cfg.normalize_feature)

    if cfg.l_loss_weight == 0:
        l_loss = 0
    elif cfg.local_dist_own_hard_sample:
        # Let local distance find its own hard samples.
        l_loss, l_dist_ap, l_dist_an, _ = local_loss(
            loss_dict['l_tri_loss'], local_feat, None, None, labels.data,
            normalize_feature=cfg.normalize_feature)
    else:
        l_loss, l_dist_ap, l_dist_an = local_loss(
            loss_dict['l_tri_loss'], local_feat, p_inds, n_inds, labels.data,
            normalize_feature=cfg.normalize_feature)


    total_loss = g_loss * cfg.g_loss_weight \
            + l_loss * cfg.l_loss_weight \

    return total_loss

