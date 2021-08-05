# PFLD Implementation with Pytorch
An open source program reference to [PFLD](https://arxiv.org/pdf/1902.10859.pdf ).


## Install
```bash
$ git clone git@github.com:yeyeck/face_landmarks.git
$ cd face_landmarks
$ pip install -r requirements.txt
```
## Datasets
It provide a method to load [WFLW](https://wywu.github.io/projects/LAB/WFLW.html) datasets which contains 98 face landmarks. WFLW contains 10000 samples, 7500 for training and 2500 for testing. The dataset folder should like:
```bash
|--wflw_dataset
    |--images
        |--0--Parade
        |--1--Handshaking
        |...
    --train.txt
    --test.txt
```

## Training
```bash
$ python train.py --data <path of data>
```
Some other parameters
|param|default|description|
|--|--|--|
|--loss|WING|optional: WING, MSE or SMOOTHL1|
|--epochs|100|epochs|
|--lr|0.01|learning rate|
|--img-size|128|input size|
|--optimizer|sgd|optional: adam or sgd|
|--name|exp|directory name to save results|
|--bath-size|256||