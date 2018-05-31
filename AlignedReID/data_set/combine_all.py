#-*- coding:utf-8 -*-
#===================================
#combine all data set together
#===================================
from __future__ import print_function

import os.path as osp

ospeu = osp.expanduser
ospj = osp.join
ospap = osp.abspath

import numpy as np
from collections import defaultdict
import shutil

import sys
sys.path.append('../')

from reid_utils.common_utils import may_make_dir
from reid_utils.common_utils import save_pickle
from reid_utils.common_utils import load_pickle

from reid_utils.dataset_utils import new_im_name_tmpl
from reid_utils.dataset_utils import parse_full_path_new_im_name
from reid_utils.dataset_utils import parse_original_new_im_name
from reid_utils.dataset_utils import parse_original_duke_im_name
from reid_utils.dataset_utils import parse_original_market1501_im_name
from reid_utils.dataset_utils import parse_original_msmt17_im_name
from reid_utils.dataset_utils import parse_original_msmt17_im_name
from reid_utils.dataset_utils import partition_train_val_set


def move_ims(
        ori_im_paths,
        new_im_dir,
        parse_im_name,
        new_im_name_tmpl,
        new_start_id):
    """Rename and move images to new directory."""
    ids = [parse_im_name(osp.basename(p), 'id') for p in ori_im_paths]
    cams = [parse_im_name(osp.basename(p), 'cam') for p in ori_im_paths]

    unique_ids = list(set(ids))
    unique_ids.sort()
    id_mapping = dict(
        zip(unique_ids, range(new_start_id, new_start_id + len(unique_ids))))

    new_im_names = []
    cnt = defaultdict(int)
    for im_path, id, cam in zip(ori_im_paths, ids, cams):
        new_id = id_mapping[id]
        cnt[(new_id, cam)] += 1
        new_im_name = new_im_name_tmpl.format(new_id, cam, cnt[(new_id, cam)] - 1)
        shutil.copy(im_path, ospj(new_im_dir, new_im_name))
        new_im_names.append(ospj(new_im_dir, new_im_name))
    return new_im_names, id_mapping



def get_parse_im_funtion(data_set):
    if data_set == 'market1501':
        return parse_original_market1501_im_name
    elif data_set =='cuhk03':
        return parse_original_new_im_name
    elif data_set =='duke':
        return parse_original_duke_im_name
    elif data_set =='msmt17':
        return parse_original_msmt17_im_name

def combine_trainval_sets(
        im_dirs,
        partition_files,
        data_sets,
        save_dir):
    new_im_dir = ospj(save_dir, 'trainval_images')
    may_make_dir(new_im_dir)
    new_im_names = []
    new_start_id = 0
    for im_dir, partition_file, data_set in zip(im_dirs, partition_files, data_sets):
        parse_im_name = get_parse_im_funtion(data_set)
        partitions = load_pickle(partition_file)
        im_paths = [ospj(im_dir, n) for n in partitions['trainval_im_names']]
        im_paths.sort()
        new_im_names_, id_mapping = move_ims(
            im_paths, new_im_dir, parse_im_name, new_im_name_tmpl, new_start_id)
        new_start_id += len(id_mapping)
        new_im_names += new_im_names_

    trainval_ids = range(new_start_id)
    
    partitions = partition_train_val_set(
                    new_im_names, parse_full_path_new_im_name, val_ids_num=300)

    train_im_names = partitions['train_im_names']
    train_ids = list(set([parse_full_path_new_im_name(n, 'id')
                        for n in partitions['train_im_names']]))
    train_ids.sort()
    train_ids2labels = dict(zip(train_ids, range(len(train_ids))))

    val_marks = [0, ] * len(partitions['val_query_im_names']) \
                + [1, ] * len(partitions['val_gallery_im_names'])
    val_im_names = list(partitions['val_query_im_names']) \
                    + list(partitions['val_gallery_im_names'])

    partitions = {'trainval_im_names': new_im_names,
                    'trainval_ids2labels': dict(zip(trainval_ids, trainval_ids)),
                    'train_im_names': train_im_names,
                    'train_ids2labels': train_ids2labels,
                    'val_im_names': val_im_names,
                    'val_marks': val_marks,
                    }
    partition_file = ospj(save_dir, 'partitions.pkl')
    save_pickle(partitions, partition_file)
    print('Partition file saved to {}'.format(partition_file))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine Trainval Set of Market1501, CUHK03, DukeMTMC-reID")

    # Image directory and partition file of transformed datasets
    parser.add_argument(
        '--market1501_im_dir',
        type=str,
        default=ospeu('/data/DataSet/market1501/')
    )
    parser.add_argument(
        '--market1501_partition_file',
        type=str,
        default=ospeu('/data/DataSet/market1501/partitions.pkl')
    )

    cuhk03_im_type = ['detected', 'labeled'][0]
    parser.add_argument(
        '--cuhk03_im_dir',
        type=str,
        # Remember to select the detected or labeled set.
        default=ospeu('/data/DataSet/cuhk03/{}/images'.format(cuhk03_im_type))
    )
    parser.add_argument(
        '--cuhk03_partition_file',
        type=str,
        # Remember to select the detected or labeled set.
        default=ospeu('/data/DataSet/cuhk03/{}/partitions.pkl'.format(cuhk03_im_type))
    )

    parser.add_argument(
        '--duke_im_dir',
        type=str,
        default=ospeu('/data/DataSet/duke/'))
    parser.add_argument(
        '--duke_partition_file',
        type=str,
        default=ospeu('/data/DataSet/duke/partitions.pkl')
    )

    parser.add_argument(
        '--msmt17_im_dir',
        type=str,
        default=ospeu('/data/DataSet/msmt17/'))
    parser.add_argument(
        '--msmt17_partition_file',
        type=str,
        default=ospeu('/data/DataSet/msmt17/partitions.pkl')
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default=ospeu('/data/DataSet/combine')
    )

    args = parser.parse_args()

    im_dirs = [
        ospap(ospeu(args.market1501_im_dir)),
        ospap(ospeu(args.cuhk03_im_dir)),
        ospap(ospeu(args.duke_im_dir)),
        ospap(ospeu(args.msmt17_im_dir))
    ]
    partition_files = [
        ospap(ospeu(args.market1501_partition_file)),
        ospap(ospeu(args.cuhk03_partition_file)),
        ospap(ospeu(args.duke_partition_file)),
        ospap(ospeu(args.msmt17_partition_file))
    ]
    data_sets = ['market1501', 'cuhk03', 'duke', 'msmt17']

    save_dir = ospap(ospeu(args.save_dir))
    for data_set in data_sets:
        save_dir = save_dir + '_' + data_set
    may_make_dir(save_dir)

    combine_trainval_sets(im_dirs, partition_files, data_sets, save_dir)
