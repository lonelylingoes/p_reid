Implementation for CVPR 2017 Paper: "AlignedReID: Surpassing Human-level Performance in Person Re-Identification" by pytorch.


Thanks for [huanghoujing's AlignedReID-Re-Production-Pytorch](https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch), I completed the Implementation of the paper. I rewrited some codes for easier understanding to myself, and there are many similarities in the repository with huanghoujing's.

If you adopt AlignedReID in your research, please cite the paper

```
@article{zhang2017alignedreid,
  title={AlignedReID: Surpassing Human-Level Performance in Person Re-Identification},
  author={Zhang, Xuan and Luo, Hao and Fan, Xing and Xiang, Weilai and Sun, Yixiao and Xiao, Qiqi and Jiang, Wei and Zhang, Chi and Sun, Jian},
  journal={arXiv preprint arXiv:1711.08184},
  year={2017}
}
```
---
## Dataset Preparation
1. download the dataset.
2. run the script 
```
cd AlignedReID/data_set
python market1501_prepare.py
```
this scrpts will prepare the market1501 dataset, and you can prepare other datasets by runing other scripts, or run the script 'combine_all.py' to combine them togather.


## training script
#### ResNet-50 + Global Loss + Local loss on Market1501
train the single model on market1501 from scratch with global and local loss use this script.
```
cd AlignedReID/script

python train.py \
--train_dataset market1501 \
--train_partitons /data/DataSet/market1501/partitions.pkl \
--ids_per_batch 32 \
--ims_per_id 4 \
--normalize_feature false \
-gm 0.3 \
-lm 0.3 \
-glw 1 \
-llw 0 \
-idlw 0 \
--base_lr 2e-4 \
--lr_decay_type exp \
--exp_decay_at_epoch 151 \
--total_epochs 300
```
train from checkpoint use this script.

```
cd AlignedReID/script

python train.py \
--resume True
--train_dataset market1501 \
--train_partitons /data/DataSet/market1501/partitions.pkl \
--ids_per_batch 32 \
--ims_per_id 4 \
--normalize_feature false \
-gm 0.3 \
-lm 0.3 \
-glw 1 \
-llw 0 \
-idlw 0 \
--base_lr 2e-4 \
--lr_decay_type exp \
--exp_decay_at_epoch 151 \
--total_epochs 300
```

## testing script
for testing, you can run this script.   

```
cd AlignedReID/script

python test.py \
--onlytest True \
--to_re_rank True \
--train_dataset market1501 \
--test_dataset market1501 \
--test_partitons /data/DataSet/market1501/partitions.pkl \
```
if you want to test on other dataset, use this script.

```
cd AlignedReID/script

python test.py \
--onlytest True \
--to_re_rank True \
--train_dataset market1501 \
--test_dataset duke \
--test_dataset_partitions /data/DataSet/duke/partitions.pkl
```

## results
On Market1501 with setting
- Train only on Market1501 (While the paper combines 4 datasets.)
-  NOT normalizing feature to unit length, with margin 0.3
- Adam optimizer, base learning rate 2e-4, decaying exponentially after 150 epochs. Train for 300 epochs in total.

|   | Rank-1 (%) | mAP ( %) | Rank-1 (%) after Re-ranking | mAP (%) after Re-ranking |
| --- | :---: | :---: | :---: | :---: |
| train with global + local loss/test with global distance | 88.54 | 73.62 | 90.68 | 86.63 |
| train with global + local + identify loss/test with global distance | 85.84 | 68.85 | 87.56 | 81.35 |

## further sesults
Use some advices from 'Re-ID done right：towards good practices for person re-identification', the finnal results is:

|   | Rank-1 (%) | mAP ( %) | Rank-1 (%) after Re-ranking | mAP (%) after Re-ranking |
| --- | :---: | :---: | :---: | :---: |
| train with global + local loss/test with global distance | 91.09 | 78.17 | 92.73 | 89.82 |
