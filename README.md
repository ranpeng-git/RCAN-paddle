# Image Super-Resolution Using Very Deep Residual Channel Attention Networks

This repository is paddle implements for RCAN introduced in the following paper

> [Yulun Zhang](http://yulunzhang.com/), [Kunpeng Li](https://kunpengli1994.github.io/), [Kai Li](http://kailigo.github.io/), [Lichen Wang](https://sites.google.com/site/lichenwang123/), [Bineng Zhong](https://scholar.google.de/citations?user=hvRBydsAAAAJ&hl=en), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/), "Image Super-Resolution Using Very Deep Residual Channel Attention Networks", ECCV 2018, [[arXiv]](https://arxiv.org/abs/1807.02758) 



## Contents

1. [Introduction](#introduction)

2. [Train](#train)
3. [Test](#test)
4. [Results](#results)



## Introduction

Convolutional neural network (CNN) depth is of crucial importance for image super-resolution (SR). However, we observe that deeper networks for image SR are more difficult to train. The low-resolution inputs and features contain abundant low-frequency information, which is treated equally across channels, hence hindering the representational ability of CNNs. To solve these problems, we propose the very deep residual channel attention networks (RCAN). Specifically, we propose a residual in residual (RIR) structure to form very deep network, which consists of several residual groups with long skip connections. Each residual group contains some residual blocks with short skip connections. Meanwhile, RIR allows abundant low-frequency information to be bypassed through multiple skip connections, making the main network focus on learning high-frequency information. Furthermore, we propose a channel attention mechanism to adaptively rescale channel-wise features by considering interdependencies among channels. Extensive experiments show that our RCAN achieves better accuracy and visual improvements against state-of-the-art methods.

<img src="figs/CA.PNG" width="900px" height="200px"/>



<img src="figs/RCAB.PNG" width="900px" height="200px"/>

<img src="figs/RCAN.PNG" width="900px" height="300px"/>

The architecture of our proposed residual channel attention network (RCAN).



## Train

###  Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

2. Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

### Begin to train

1. [Download the initialized model with code `1p89`](https://pan.baidu.com/s/1ThXJKouEXiGn0SBVWODm0w) and place them in '/RCAN_TrainCode/model'.

2. Cd to 'RCAN_TrainCode/code', run the following scripts to train models.

**You can use scripts in file 'TrainRCAN_scripts' to train models for our paper.**

```
python main.py --model RCAN --save RCAN_X4 --scale 4 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --patch_size 192 --pre_train model/model_init.pdparams  2>&1 | tee $LOG
```



## Test

### Quick start

1. Download [models of ours with code: `3d66`]( https://pan.baidu.com/s/1GJGHWdeiTKhathBGtctTuw)  and place them in '/RCAN_TrainCode/model'

2. Cd to '/RCAN_TrainCode/code', run the following scripts.

​    **You can use scripts in file 'TestRCAN_scripts' to produce results for our paper.**

```
python main.py --data_test MyImage --scale 4 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64  --test_only --save_results --chop --self_ensemble --save 'RCAN_test' --testpath  --testset  --pre_train model/model_225.pdparams
```



### The whole test pipeline

1. Prepare test data. Place the original test sets (e.g., Set14, other test sets are available from [GoogleDrive](https://drive.google.com/drive/folders/1xyiuTr6ga6ni-yfTP7kyPHRmfBakWovo?usp=sharing) or [Baidu](https://pan.baidu.com/s/1yBI_-rknXT2lm1UAAB_bag)) in 'OriginalTestData'.
2. Run 'Prepare_TestData_HR_LR.m' in Matlab to generate HR/LR images with different degradation models.
3. Conduct image SR. See **Quick start**.
4. Evaluate the results.
5. Run 'Evaluate_PSNR_SSIM.m' to obtain PSNR/SSIM values for paper. Or using `/RCAN_TrainCode/metric.py` to obtain PSNR/SSIM.



## Results

### Quantitative Results

|     method     |         Set15          | Epoch |
| :------------: | :--------------------: | :---: |
| pytorch RCAN++ | PSNR:28.98 SSIM:0.7901 | 1000  |
| Paddle RCAN++  | PSNR:26.82 SSIM:0.7723 |  225  |

### Visiual outcome

<center class="half">
  <img src="figs/baboon_LRBI_x4_x4_SR.png"  width="200px" height="200px"/>
  <img src="figs/barbara_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/bridge_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/coastguard_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/comic_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/face_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/flowers_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/foreman_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/lenna_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/man_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/monarch_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/pepper_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/ppt3_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
  <img src="figs/zebra_LRBI_x4_x4_SR.png" width="200px" height="200px"/>
<center />

