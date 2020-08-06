# PRDNet
PRDNet: Medical image segmentation based on parallel residual and dilated network

## Statement
This code is borrowed from [Multi-Scale-Attention](https://github.com/sinAshish/Multi-Scale-Attention), our contributions are ```prdnet.py```, ```test.py``` and ```inference.py```

## Environment
Our model is tested in the following environment:   
  * python3.5(anaconda3)
  * pytoch1.2.0
  * torchvision0.4.0
  * CUDA10.0
  * GPU:Nvidia 1080Ti(11G)
  * Visdom


## A quick demo
1. You can download our pretrained model [Best_MR2.pth](https://pan.baidu.com/s/1SToITGqAHMPrTLqrGiq5YQ) from BaiduNetDisk. The extraction code is ```PRDN```.
2. Put the downloaded pretrained model in the ```./model``` folder.
3. Run ```python inference.py``` to realize a quick demo.

## Training
1. CHAOS dataset can be downloaded from its official [website](https://chaos.grand-challenge.org/).
2. The processing of the original dataset can be referred to "[prepare your data](https://github.com/sinAshish/Multi-Scale-Attention)".
3. Once you've divided the training set, the validation set and the test set. You should place the groundtruth of the validation set and its corresponding DCM format images in the corresponding folder. 
4. We have placed our divided groundtruth and its corresponding DCM format images under the "Data_3D" folder. You can replace the contents with yours.
5. Run ```python train.py```

## Testing
1. Preparing test set as mentioned in [Training](#Training).
2. Run ```python test.py``` 
