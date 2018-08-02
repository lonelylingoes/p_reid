#-*- coding:utf-8 -*-
#===================================
# deploy program
#===================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from sklearn.utils.linear_assignment_ import linear_assignment
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from AlignedReID.model.model import Model
import AlignedReID.reid_utils.common_utils as common_utils 
import AlignedReID.reid_utils.model_utils as model_utils
from AlignedReID.reid_utils.model_utils import transer_var_tensor
import  AlignedReID.model.loss as loss
from AlignedReID.reid_utils.common_utils import measure_time
from AlignedReID.reid_utils.re_ranking import re_ranking
from person import Status


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



class ReId(object):
    '''
    the class is created for deloy purpose.
    '''
    
    def __init__(self, 
                model_path,
                identy_threshold,
                device_id = 0):
        '''
        args:
            model_path: the model file path
            image_path: the image file path
        '''
        class Config(object):
            pass
        cfg = Config
        cfg.model_weight_file = ''
        cfg.ckpt_file = model_path
        self.device_id = device_id
        self.identy_threshold = identy_threshold
        torch.cuda.set_device(device_id)
        # create model
        self.model = Model(local_conv_out_channels=128, pretrained = False)
        # load model param
        self.model = model_utils.load_test_model(self.model, cfg)
        self.model.eval()
        # after load model
        if torch.cuda.is_available() and device_id >= 0:
            self.model = self.model.cuda(device = device_id)
    
    def __parse_image_name__(self, image_name):
        '''
        parse the image name to mark, person id, camera id, scene id, 
        args:
            image_name: the name of the image
        '''
        mark = int(image_name[0])
        person_id = int(image_name[2:6])
        camera_id = int(image_name[8])
        scene_id = int(image_name[10])
        return mark, person_id, camera_id, scene_id

    def get_threshold(self):
        return self.identy_threshold

    def extract_features(self, pictures):
        '''
        only extract global features.
        args:
            pictures: the image array list
        returns:
            features: the features vector extracted by the model, the shape is (n,c)
        '''
        pictures = [common_utils.pre_process_im(picture, (416, 208)) for picture in pictures]
        pictures = np.array(pictures)
        with torch.no_grad():
            ims_var = Variable(transer_var_tensor(torch.from_numpy(pictures), self.device_id).float())
        global_feats, local_feats = self.model(ims_var)[:2]
        global_feats = global_feats.data.cpu().numpy()
        return global_feats


    @staticmethod
    def association_judge(
                        unconfirmed_persons,
                        confirmed_persons,
                        threshold,
                        identy_confirm_strategy = 'min',
                        to_re_rank=False):
        '''
        judge the association between unconfirmed and confirmed person.
        the memory out problem was not conserdered.
        args:
            unconfirmed_persons: confirmed persons list
            confirmed_persons: unconfirmed persons list
            threshold: identy threshold decides whether same or not
            identy_confirm_strategy: value in 'average', 'max', 'min', for identiy
            to_re_rank: whether use re_rank
        returns: list of person number, if not found the value is -1.
        '''
        # if found, fill the id else fill -1
        found_ids = [-1 for i in unconfirmed_persons]
        if len(confirmed_persons) == 0:
            return found_ids
        confirmed_persons = sorted(confirmed_persons, key=lambda person: person.status)
        try:
            confirmed_index = [person.status for person in confirmed_persons].index(Status.confirmed)
        except:
            confirmed_index = -1
        compact_q_g_dist = ReId.get_associate_dis(unconfirmed_persons,confirmed_persons,
                                            identy_confirm_strategy, to_re_rank)
        
        # associate confirmed person first 
        if confirmed_persons != -1:
            m, n = compact_q_g_dist.shape
            for q in range(m):
                if np.min(compact_q_g_dist[q,confirmed_index:]) < threshold:
                    compact_q_g_dist[q,:confirmed_index] = threshold + 1e-5
        # Solve the linear assignment problem using the Hungarian algorithm.
        indices = linear_assignment(compact_q_g_dist)
        for row, col in indices:
            if compact_q_g_dist[row, col] < threshold:
                found_ids[row] = confirmed_persons[col].person_number
        
        
        '''
        # the numberof query is m, the number of gallery is n
        m, n = compact_q_g_dist.shape
        # sort and find correct matches
        indexs = np.argmin(compact_q_g_dist, axis=1)
        sorted_dis = np.sort(compact_q_g_dist, axis =1)
        # judge for every query
        for i in range(m):
            # for query i, in the gallery set, the shortest distance is less than the threshold
            if sorted_dis[i][0] < threshold:
                found_ids[i]=confirmed_persons[indexs[i]].person_number
        found_ids = self.__remvoe_overlap__(sorted_dis[:,0], found_ids)
        '''

        return found_ids



    @staticmethod
    def get_associate_dis(unconfirmed_persons,
                        confirmed_persons,
                        identy_confirm_strategy = 'min',
                        to_re_rank=False):
        '''
        judge the association between unconfirmed and confirmed person.
        the memory out problem was not conserdered.
        args:
            unconfirmed_persons: confirmed persons list
            confirmed_persons: unconfirmed persons list
            identy_confirm_strategy: value in 'average', 'max', 'min', for identiy
            to_re_rank: whether use re_rank
        returns: distance matrix
        '''      
        q_feats = np.array([person.get_last_tracking_feature() for person in unconfirmed_persons])
        # get gallery features, person numbers and different person indexes list.
        g_feats = []
        index = 0
        index_list = [index]
        person_numbers = []
        confirmed_strategy = []
        for person in confirmed_persons:
            person_numbers.append(person.person_number)
            cache_len = person.get_identy_cache_len() + person.get_tracking_cache_len()
            index += cache_len
            index_list.append(index)
            cache_info = []
            cache_info.extend(person.get_tracking_info())
            cache_info.extend(person.get_identy_info())
            confirmed_strategy.append(identy_confirm_strategy)
            for i in range(cache_len):
                g_feats.append(cache_info[i].body_feature)
        g_feats = np.array(g_feats)

        # compute distance
        q_g_dist = loss.compute_dist_np(q_feats, g_feats, type='euclidean')
        compact_q_g_dist = ReId.compact_dist(q_g_dist, index_list, confirmed_strategy)
        
        return compact_q_g_dist



    @staticmethod
    def compact_dist(g_q_dist, index_list, compact_strategy):
        '''
        compact the distance matrix by the pointed strategy.
        the identy distance has higher priority, 
        for easy computation, add the delta distance, and use identy threshold.
        args:
            g_q_dist: orinal matrix.
            index_list: the index partion list.
            compact_strategy:list of 'mean', 'max', 'min' 
        '''
        compact_q_g_dist = np.zeros((g_q_dist.shape[0], len(index_list)-1))
        for i in range(len(index_list) - 1):
            if compact_strategy[i] == 'mean':
                    compact_q_g_dist[:,i] = np.mean(g_q_dist[:,index_list[i]:index_list[i+1]], axis=1)
            elif compact_strategy[i] == 'min':
                    compact_q_g_dist[:,i] = np.min(g_q_dist[:,index_list[i]:index_list[i+1]], axis=1)
            elif compact_strategy[i] == 'max':
                    compact_q_g_dist[:,i] = np.max(g_q_dist[:,index_list[i]:index_list[i+1]], axis=1)
            else:
                compact_q_g_dist[:,i] = np.mean(g_q_dist[:,index_list[i]:index_list[i+1]], axis=1)

        return compact_q_g_dist


    def judge_from_file(self, 
                images_path,
                to_re_rank=True,
                use_local_distance=False,
                normalize_feature = False):
        '''
        judge the querys wether asoociate with the gallerys which are fixed person pictures
        args:
            images_path: the path of images
            to_re_rank: whether use re_rank
            use_local_distance: whether use local distance
            normalize_feature: whether normalize the features
        returns:
            found_ids:

        '''
        # get images infomation
        images, marks, person_ids, camera_ids, scene_ids = self.__get_images_info__(images_path)
        # get query and gallery indicate
        q_inds = (marks == 0)
        g_inds = (marks == 1)
        # get the distance matrix
        dist_mat= self.__compute_distance_mat__(images, marks, q_inds, g_inds,
                                                to_re_rank, use_local_distance, normalize_feature)
        # query_ids = person_ids[q_inds]
        gallery_ids = person_ids[g_inds]
        # if found, fill the id else fill -1
        found_ids = []
        # the numberof query is m, the number of gallery is n
        m, n = dist_mat.shape
        # the threshold decides whether same
        threshold = self.identy_threshold
        # sort and find correct matches
        indexs = np.argmin(dist_mat, axis=1)
        sorted_dis = np.sort(dist_mat, axis =1)

        # judge for every query
        for i in range(m):
            # for query i, in the gallery set, the shortest distance is less than the threshold
            if sorted_dis[i][0] < threshold:
                found_ids.append(gallery_ids[indexs][i])
            else:
                found_ids.append(-1)

        found_ids = self.__remvoe_overlap__(sorted_dis[:,0], found_ids)
        return found_ids

    def __remvoe_overlap__(self, sorted_dis_vect, found_ids):
        '''
        for every query remove the overlap gallery id by compare the distances
        args:
            sorted_dis_vect: for very for every query the shortest distance vector,numpy array
            found_ids: for very query denote whether found, list     
        '''
        found_array = np.array(found_ids)
        # for all item in gallery_ids
        for ids in found_ids:
            if ids == -1:
                continue
            index = np.argwhere(found_array == ids)
            if len(index) > 1:#find overlap
                cut_vect = sorted_dis_vect[index]
                shortest_dis_index = np.argmin(cut_vect)
                for i in range(len(index)):
                    if i == shortest_dis_index:
                        continue
                    found_ids[index[i][0]] = -1
        return found_ids

    def __get_images_info__(self, images_path):
        '''
        read detected images and base images from 'images_path',
        and return images arrary, marks, person ids, camera ids, scene ids, 
        '''
        images=[]
        marks=[]
        person_ids=[]
        camera_ids=[]
        scene_ids=[]
        files = os.listdir(images_path)
        files.sort()
        for file in files:
            mark, person_id, camera_id, scene_id = self.__parse_image_name__(file)
            images.append(common_utils.pre_process_im(os.path.join(images_path, file), (416, 208)))
            marks.append(mark)
            person_ids.append(person_id)
            camera_ids.append(camera_id)
            scene_ids.append(scene_id)
        images = np.array(images)
        marks = np.array(marks)
        person_ids = np.array(person_ids)
        camera_ids = np.array(camera_ids)
        scene_ids = np.array(scene_ids)

        return images, marks, person_ids, camera_ids, scene_ids

    def __compute_distance_mat__(self, 
                            images,
                            marks,
                            q_inds,
                            g_inds,
                            to_re_rank,
                            use_local_distance,
                            normalize_feature):
        '''
        compute distance mat
        args:
            images: the numpy arrary of images
            marks: the numpy arrary of marks denote query images or gallery images
            q_inds: query images indicates
            g_inds: gallery images indicates
            to_re_rank: whether use re_rank
            use_local_distance: whether use local distance
            normalize_feature: whether normalize the features
        returns:
            the finnal distance matrix
        '''
        with measure_time('Extrating feature...'):
            ims_var = Variable(transer_var_tensor(torch.from_numpy(images), self.device_id).float(), volatile=True)
            global_feats, local_feats = self.model(ims_var)[:2]
            global_feats = global_feats.data.cpu().numpy()
            local_feats = local_feats.data.cpu().numpy()

        if normalize_feature:
            global_feats = loss.normalize_np(global_feats, axis=1)
            local_feats = loss.normalize_np(local_feats, axis=-1)

        # Global Distance 
        with measure_time('Computing global distance...'):
            # query-gallery distance using global distance
            global_q_g_dist = loss.compute_dist_np(
                global_feats[q_inds], global_feats[g_inds], type='euclidean')

        if to_re_rank:
            with measure_time('Re-ranking...'):
                # query-query distance using global distance
                global_q_q_dist = loss.compute_dist_np(
                    global_feats[q_inds], global_feats[q_inds], type='euclidean')
                # gallery-gallery distance using global distance
                global_g_g_dist = loss.compute_dist_np(
                    global_feats[g_inds], global_feats[g_inds], type='euclidean')
                # re-ranked global query-gallery distance
                re_global_q_g_dist = re_ranking(
                    global_q_g_dist, global_q_q_dist, global_g_g_dist)

        # Local Distance 
        if use_local_distance:
            with measure_time('Computing local distance...'):
                # query-gallery distance using local distance
                local_q_g_dist = low_memory_local_dist(
                    local_feats[q_inds], local_feats[g_inds])
            if to_re_rank:
                with measure_time('Re-ranking...'):
                    # query-query distance using local distance
                    local_q_q_dist = low_memory_local_dist(
                        local_feats[q_inds], local_feats[q_inds])
                    # gallery-gallery distance using local distance
                    local_g_g_dist = low_memory_local_dist(
                        local_feats[g_inds], local_feats[g_inds])

            # Global+Local Distance 
            global_local_q_g_dist = global_q_g_dist + local_q_g_dist
            if to_re_rank:
                with measure_time('Re-ranking...'):
                    global_local_q_q_dist = global_q_q_dist + local_q_q_dist
                    global_local_g_g_dist = global_g_g_dist + local_g_g_dist
                    re_global_local_q_g_dist = re_ranking(
                        global_local_q_g_dist, global_local_q_q_dist, global_local_g_g_dist)

        # return distance
        if use_local_distance:
            if to_re_rank:
                return re_global_local_q_g_dist
            else:
                return global_local_q_g_dist
        else:
            if to_re_rank:
                return re_global_q_g_dist
            else:
                return global_q_g_dist



def main():
    reId = ReId('/data/chensijing/AlignedReID/ckpt_dir/ckpt_path', 1.5)
    for i in range(10):
        found_ids = reId.judge_from_file('/home/ubun-titan/Debug/image_dir1')
    print(found_ids)

if __name__ == '__main__':
    main()