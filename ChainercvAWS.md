# Training SSD <br>with ChainerCV on AWS

<br>

## Preparation

### Log in to your Deep Learning AMI (Ubuntu), instance type = p2.xlarge


Create a conda environment

```
conda create -n chainer python=3.6 anaconda
source activate chainer
```

### Install Chainer

```
pip install -U numpy
pip install cupy
pip install chainer
pip install chainercv
```

### Install Libraries
```
conda install -c https://conda.anaconda.org/menpo opencv3
pip install -U numpy
pip install matplotlib
pip install pillow
sudo apt-get install zip unzip
sudo apt-get -y install imagemagick
```

### Where is Data Path ?

* ChainerCV Install Path
```
home/ubuntu/anaconda3/envs/chainer/lib/python3.6/site-packages
```

* Expanded Datasets
```
/home/ubuntu/.chainer/dataset/_dl_cache/
/home/ubuntu/.chainer/dataset/pfnet/chainercv/voc/
```

<br>

### Start Training


```
cd chainercv/example/ssd
python train.py
```


### Download data using scp

Download data from server to local
```
scp -i (keyfilename).pem ubuntu@52.40.30.xxx:/home/ubuntu/chainercv/examples/ssd/image/*.png .
```

Upload data from local to server
```
scp -i (keyfilename).pem ./*.zip ubuntu@52.40.30.xxx:/home/ubuntu/
```

Please set your private key file name.


### Train SSD with your own dataset

image-labelling-tool
https://github.com/yuyu2172/image-labelling-tool/tree/master/examples/ssd
