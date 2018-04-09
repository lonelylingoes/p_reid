Implementation for CVPR 2017 Paper: "Re-ID done right: towards good practices for person re-identiﬁcation" by pytorch.


If you adopt AlignedReID in your research, please cite the paper

```
@article{
  title={Re-ID done right: towards good practices for person re-identiﬁcation},
  author={Jon Almaza´n, Bojana Gajic´, Naila Murray, Diane Larlus},
  journal={arXiv preprint arXiv:1801.05339v1},
  year={2018}
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
--dataset market1501 \
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
--dataset market1501 \
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
--dataset market1501 \
```
if you want to test on other dataset, use this script.

```
cd AlignedReID/script

python test.py \
--onlytest True \
--to_re_rank True \
--dataset duke \
--dataset_partitions /data/DataSet/duke/partitions.pkl
```

## results
On Market1501 with setting
- Train only on Market1501 (While the paper combines 4 datasets.)
-  NOT normalizing feature to unit length, with margin 0.3
- Adam optimizer, base learning rate 2e-4, decaying exponentially after 150 epochs. Train for 300 epochs in total.

|   | Rank-1 (%) | mAP ( %) | Rank-1 (%) after Re-ranking | mAP (%) after Re-ranking |
| --- | :---: | :---: | :---: | :---: |
| global + local loss | 88.54 | 73.62 | 90.68 | 86.63 |
| global + local + identify loss | 85.84 | 68.85 | 87.56 | 81.35 |
