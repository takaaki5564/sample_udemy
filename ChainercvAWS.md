# Training SSD <br>with ChainerCV on AWS

<br>

## Preparation

### Log in to your Deep Learning AMI (Ubuntu), instance type = p2.xlarge


### Install Chainer

```
pip install -U numpy
pip install cupy
pip install chainer
pip install chainercv
```

Download CheinrCV and setup conda environment
```
git clone https://github.com/chainer/chainercv
cd chainercv
conda env create -f environment.yml
source activate chainercv
```

### Install ChainerCV

```
pip install -e .
```

### Optional install

```
conda install -c https://conda.anaconda.org/menpo opencv3
conda install matplotlib
pip install pillow
sudo apt-get -y install imagemagick
```

<br>

