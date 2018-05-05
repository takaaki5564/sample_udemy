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

### Git clone ChainerCV sample code
```
https://github.com/chainer/chainercv.git
cd chainercv
git checkout b0f0e5a257608196a2e389d3f5a782d544bfc6e3
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

### Start Training SSD300

```
cd chainercv/example/ssd
python train.py --batchsize 8 --gpu 0 
```

### Evaluate train result

Launch another terminal.
```
cd chainercv/example/ssd
ls result
mkdir image
cp /home/ubuntu/.chainer/dataset/pfnet/chainercv/voc/VOCdevkit/VOC2007/JPEGImages/00000*.jpg ./image/
```

```
vim demo.py
```

Edit demo.py as follows

```
#plot.show()
save_image = args.image.replace(".jpg", "_out.png")
plot.savefig(save_image)
```

<br>
Change score threshold in ssd.py to check intermediate result
```
vim home/ubuntu/anaconda3/envs/chainer/lib/python3.6/site-packages/chainercv/links/model/ssd/ssd.py
```

The threshold value was changed to 0.4 in demo
```
self.score_threshold = 0.4  # originally 0.6
```


Eval trained model (Specify your pretrained model)
```
python demo.py ./image/00000x.jpg --pretrained_model ./result/model_iter_(xxxx)
```

You can write a script as follows.

```
vim eval.sh
```

```
python demo.py ./image/000001.jpg --pretrained_model ./result/model_iter_(xxxx)
python demo.py ./image/000002.jpg --pretrained_model ./result/model_iter_(xxxx)
python demo.py ./image/000003.jpg --pretrained_model ./result/model_iter_(xxxx)
python demo.py ./image/000004.jpg --pretrained_model ./result/model_iter_(xxxx)
python demo.py ./image/000005.jpg --pretrained_model ./result/model_iter_(xxxx)
python demo.py ./image/000006.jpg --pretrained_model ./result/model_iter_(xxxx)
python demo.py ./image/000007.jpg --pretrained_model ./result/model_iter_(xxxx)
python demo.py ./image/000008.jpg --pretrained_model ./result/model_iter_(xxxx)
python demo.py ./image/000009.jpg --pretrained_model ./result/model_iter_(xxxx)
```

```
bash eval.sh
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

Use image-labelling-tool

https://github.com/yuyu2172/image-labelling-tool/tree/master/examples/ssd
