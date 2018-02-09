#-*- coding:utf-8 -*-
#===================================
# test program
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')


import time
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
import utils.common_utils as common_utils 
from utils.common_utils import measure_time
import utils.model_utils as model_utils
import model.model as model 
import utils.model_utils as model_utils
from utils.model_utils import transer_var_tensor
from utils.re_ranking import re_ranking
from utils.metric import cmc, mean_ap
import  model.loss as loss


def test(val_loader, model, cfg):
    '''
    validate function
    '''
    # switch to evaluate mode
    model.eval()

    global_feats, local_feats, ids, cams, marks = \
      [], [], [], [], []
    
    with measure_time('Extracting feature...'):
        for i, (ims_, ids_, cams_, marks_) in enumerate(val_loader):
            ims_var = Variable(transer_var_tensor(torch.from_numpy(ims_).float()), volatile=True)
            global_feat, local_feat, logits = model(ims_var)
            global_feat = global_feat.data.cpu().numpy()
            local_feat = local_feat.data.cpu().numpy()

            global_feats.append(global_feat)
            local_feats.append(local_feat)
            ids.append(ids_)
            cams.append(cams_)
            marks.append(marks_)

    global_feats = np.vstack(global_feats)
    local_feats = np.concatenate(local_feats)
    ids = np.hstack(ids)
    cams = np.hstack(cams)
    marks = np.hstack(marks)
    if cfg.normalize_feature:
        global_feats = loss.normalize_np(global_feats, axis=1)
        local_feats = loss.normalize_np(local_feats, axis=-1)
    ###################################
    # measure accuracy and record loss#
    ###################################
    # query, gallery, multi-query indices
    q_inds = marks == 0
    g_inds = marks == 1
    mq_inds = marks == 2

    # Global Distance 
    with measure_time('Computing global distance...'):
        # query-gallery distance using global distance
        global_q_g_dist = loss.compute_dist_np(
            global_feats[q_inds], global_feats[g_inds], type='euclidean')

    with measure_time('Computing scores for Global Distance...'):
        mAP, cmc_scores = compute_score(global_q_g_dist, ids, cams, q_inds, g_inds, cfg)

    if cfg.to_re_rank:
        with measure_time('Re-ranking...'):
            # query-query distance using global distance
            global_q_q_dist = loss.compute_dist_np(
                global_feats[q_inds], global_feats[q_inds], type='euclidean')

            # gallery-gallery distance using global distance
            global_g_g_dist = loss.compute_dist_np(
                global_feats[g_inds], global_feats[g_inds], type='euclidean')

            # re-ranked global query-gallery distance
            re_r_global_q_g_dist = re_ranking(
                global_q_g_dist, global_q_q_dist, global_g_g_dist)

        with measure_time('Computing scores for re-ranked Global Distance...'):
            mAP, cmc_scores = compute_score(re_r_global_q_g_dist, ids, cams, q_inds, g_inds, cfg)


    # Local Distance 
    if cfg.use_local_distance:
        # query-gallery distance using local distance
        local_q_g_dist = low_memory_local_dist(
            local_feats[q_inds], local_feats[g_inds])

        with measure_time('Computing scores for Local Distance...'):
            mAP, cmc_scores = compute_score(local_q_g_dist, ids, cams, q_inds, g_inds, cfg)

        if cfg.to_re_rank:
            with measure_time('Re-ranking...'):
                # query-query distance using local distance
                local_q_q_dist = low_memory_local_dist(
                    local_feats[q_inds], local_feats[q_inds])

                # gallery-gallery distance using local distance
                local_g_g_dist = low_memory_local_dist(
                    local_feats[g_inds], local_feats[g_inds])

                re_r_local_q_g_dist = re_ranking(
                    local_q_g_dist, local_q_q_dist, local_g_g_dist)

            with measure_time('Computing scores for re-ranked Local Distance...'):
                mAP, cmc_scores = compute_score(re_r_local_q_g_dist, ids, cams, q_inds, g_inds, cfg)

        # Global+Local Distance 
        global_local_q_g_dist = global_q_g_dist + local_q_g_dist
        with measure_time('Computing scores for Global+Local Distance...'):
            mAP, cmc_scores = compute_score(global_local_q_g_dist, ids, cams, q_inds, g_inds, cfg)

        if cfg.to_re_rank:
            with measure_time('Re-ranking...'):
                global_local_q_q_dist = global_q_q_dist + local_q_q_dist
                global_local_g_g_dist = global_g_g_dist + local_g_g_dist

                re_r_global_local_q_g_dist = re_ranking(
                    global_local_q_g_dist, global_local_q_q_dist, global_local_g_g_dist)

            with measure_time('Computing scores for re-ranked Global+Local Distance...'):
                mAP, cmc_scores = compute_score(re_r_global_local_q_g_dist, ids, cams, q_inds, g_inds, cfg)

    # multi-query
    # TODO: allow local distance in Multi Query
    mq_mAP, mq_cmc_scores = None, None

    return mAP, cmc_scores, mq_mAP, mq_cmc_scores


def eval_map_cmc(
      q_g_dist,
      q_ids=None, g_ids=None,
      q_cams=None, g_cams=None,
      separate_camera_set=None,
      single_gallery_shot=None,
      first_match_break=None,
      topk=None):
    """
    Compute CMC and mAP.
    Args:
      q_g_dist: numpy array with shape [num_query, num_gallery], the 
        pairwise distance between query and gallery samples
    Returns:
      mAP: numpy array with shape [num_query], the AP averaged across query 
        samples
      cmc_scores: numpy array with shape [topk], the cmc curve 
        averaged across query samples
    """
    # Compute mean AP
    mAP = mean_ap(
      distmat=q_g_dist,
      query_ids=q_ids, gallery_ids=g_ids,
      query_cams=q_cams, gallery_cams=g_cams)
    # Compute CMC scores
    cmc_scores = cmc(
      distmat=q_g_dist,
      query_ids=q_ids, gallery_ids=g_ids,
      query_cams=q_cams, gallery_cams=g_cams,
      separate_camera_set=separate_camera_set,
      single_gallery_shot=single_gallery_shot,
      first_match_break=first_match_break,
      topk=topk)
    print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
          .format(mAP, *cmc_scores[[0, 4, 9]]))
    return mAP, cmc_scores


def compute_score(dist_mat, ids, cams, q_inds, g_inds, cfg):
    '''
    compute score
    '''
    mAP, cmc_scores = eval_map_cmc(
    q_g_dist=dist_mat,
    q_ids=ids[q_inds], g_ids=ids[g_inds],
    q_cams=cams[q_inds], g_cams=cams[g_inds],
    separate_camera_set=cfg.separate_camera_set,
    single_gallery_shot=cfg.single_gallery_shot,
    first_match_break=cfg.first_match_break,
    topk=10)
    return mAP, cmc_scores



def low_memory_local_dist(x, y):
    '''
    Args:
        x: numpy array, with shape []
        y: numpy array, with shape []
    Returns:
        dist: numpy array, with shape []
    '''
    with measure_time('Computing local distance...'):
        x_num_splits = int(len(x) / 200) + 1
        y_num_splits = int(len(y) / 200) + 1
        z = loss.low_memory_matrix_op(
                loss.local_dist_np, x, y, 0, 0, x_num_splits, y_num_splits, verbose=True)
    return z



def eval(global_feats,
        local_feats,
        ids,
        cams,
        marks,
        normalize_feat=True,
        use_local_distance=False,
        to_re_rank=True,
        pool_type='average'):
    """
    Evaluate using metric CMC and mAP.
    Args:
        global_feats:
        local_feats:
        normalize_feat: whether to normalize features before computing distance
        use_local_distance: whether to use local distance
        to_re_rank: whether to also report re-ranking scores
        pool_type: 'average' or 'max', only for multi-query case
        ids:
        cams:
        marks: 
    """
    pass

    
