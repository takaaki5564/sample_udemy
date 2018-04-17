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

* Install path of ChainerCV
home/ubuntu/anaconda3/envs/chainer/lib/python3.6/site-packages

* Download path of Datasets
/home/ubuntu/.chainer/dataset/_dl_cache/


<br>

