Implementation for CVPR 2018 Paper: "Image-Image Domain Adaptation with Preserved Self-Similarity and
Domain-Dissimilarity for Person Re-identiﬁcation" by pytorch.

If you this project in your research, please cite the paper

```
@inproceedings{image-image18,
  author    = {Weijian Deng and
               Liang Zheng and
               Qixiang Ye and
               Guoliang Kang and
               Yi Yang and
               Jianbin Jiao},
  title     = {Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity
               for Person Re-identification},
  booktitle = {CVPR},
  year      = {2018},
}
```

This code refered a lot of the [Simon4Yan's Learning-via-Translation](https://github.com/Simon4Yan/Learning-via-Translation) and [junyanz's pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).



## Prerequisites 
- python 2 or 3
- NVIDIA GPU + CUDA CuDNN
- Other requirement can install by this command:
```
pip install -r requirements.txt
```

## Train the model
If you want to train the model, run this script:
```
cd script
python train.py --dataroot /data/DataSet/ --sub_dirA market1501 --sub_dirB duke --name market_duke_spgan --model spgan --pool_size 50 --no_dropout --loadSize (572,286) --fineSize (512,256)
```
It may be rasied some exceptions about http, this is caused by the visdom.server not running, you can shutdown the show of plot:
```
python train.py --dataroot /data/DataSet/ --sub_dirA market1501 --sub_dirB duke --name market_duke_spgan --model spgan --pool_size 50 --no_dropout --loadSize (572,286) --fineSize (512,256) --display_id 0
```
or start the visdom.server first:
```
python -m visdom.server
```


## view the training results and loss plots
To view training results and loss plots, run python -m visdom.server and click the URL http://localhost:8097. To see more intermediate results, check out ./checkpoints/market_duke_spgan/web/index.html


## Test the model
When the model is trained, use this script to trans A to B:

```
cd script
python test.py --dataroot /data/DataSet/ --sub_dirA market1501 --sub_dirB duke --name market_duke_spgan --model test --dataset_mode single --results_dir ../results/market_to_duke --no_dropout --fineSize (512,256) --resize_or_crop scale_width
```
And use the script to trans B to A:
```
cd script
python test.py --dataroot /data/DataSet/ --sub_dirA market1501 --sub_dirB duke --name market_duke_spgan --model test --dataset_mode single --results_dir ../results/duke_to_market --no_dropout --fineSize (512,256) --resize_or_crop scale_width --which_direction BtoA
```

When finish the test, you can watch the trans results through the webpage by open the 'index.html' in the result dir.



