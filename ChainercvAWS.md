# Training SSD <br>with ChainerCV on AWS

<br>

## Preparation

Log in to your Deep Learning AMI (Ubuntu), instance type = p2.xlarge


Create a conda environment

```
conda create -n chainer python=3.6 anaconda
source activate chainer
```

Install Chainer

```
pip install -U numpy
pip install cupy
pip install chainer
pip install chainercv
```

Install Related Libraries

```
conda install -c https://conda.anaconda.org/menpo opencv3
pip install matplotlib
pip install pillow
sudo apt-get install zip unzip
```

Get sample code of ChainerCV

```
https://github.com/chainer/chainercv.git
cd chainercv
git checkout b0f0e5a257608196a2e389d3f5a782d544bfc6e3
```

ChainerCV Install Path
```
/home/ubuntu/anaconda3/envs/chainer/lib/python3.6/site-packages
```

Expanded Train Datasets
```
/home/ubuntu/.chainer/dataset/_dl_cache/
/home/ubuntu/.chainer/dataset/pfnet/chainercv/voc/
```

<br>

## Train SSD-300 with VOC Dataset

Edit train.py
```
cd ~/chainercv/example/ssd
vim train.py
```

```
:
def main():
    :
    #train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    :
    trainer.extend(
        #extensions.ExponentialShift('lr', 0.1, init=1e-3),
        extensions.ExponentialShift('lr', 0.1, init=5e-4),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))
    
    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=voc_bbox_label_names),
        #trigger=(10000, 'iteration'))    
        trigger=(2000, 'iteration'))  
        :
    #trainer.extend(extensions.snapshot(), trigger=(10000, 'iteration'))
    trainer.extend(extensions.snapshot(), trigger=(2000, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        #trigger=(120000, 'iteration'))
        trigger=(2000, 'iteration'))
        :
```

Run train.py
```
python train.py --batchsize 8 --gpu 0 
```
<br>

## Evaluate Train Result

Launch another terminal and log in to AWS server for evaluation

Check result and copy images for evaluation 

```
cd chainercv/example/ssd
ls result
mkdir image
cp /home/ubuntu/.chainer/dataset/pfnet/chainercv/voc/VOCdevkit/VOC2007/JPEGImages/00000*.jpg ./image/
```

Edit demo.py to save result images

```
vim demo.py
```

demo.py was modified as follows

```
#plot.show()
save_image = args.image.replace(".jpg", "_out.png")
plot.savefig(save_image)
```

(Option) Modify score threshold in ssd.py to check intermediate result
```
vim /home/ubuntu/anaconda3/envs/chainer/lib/python3.6/site-packages/chainercv/links/model/ssd/ssd.py
```

Threshold value was changed as follows

```
self.score_threshold = 0.4  # 0.6 originally
```


Run demo.py with the trained model

```
python demo.py ./image/00000x.jpg --pretrained_model ./result/model_iter_(xxxx)
```

Download data from server using scp

```
scp -i (keyfilename).pem ubuntu@52.40.30.xxx:/home/ubuntu/chainercv/examples/ssd/image/*.png .
```

(Option) Write evaluation script as follows

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


(Option) Upload data to server using scp 

```
scp -i (keyfilename).pem ./*.zip ubuntu@52.40.30.xxx:/home/ubuntu/
```
<br>

## (Option) Re-configure SSD300 network

Edit network configuration

```
cd /home/ubuntu/anaconda3/envs/chainer/lib/python3.6/site-packages/chainercv
vim links/model/ssd/ssd_vgg16.py
```

e.g. Reduce number of bounding box

```
class VGG16(chainer.Chain):
        :
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        #ys.append(self.norm4(h))
        
class VGG16Extractor300(VGG16):
    :
    #grids = (38, 19, 10, 5, 3, 1)
    grids = (19, 10, 5, 3)
            :
            #self.conv11_1 = L.Convolution2D(128, 1, **init)
            #self.conv11_2 = L.Convolution2D(256, 3, **init)
        :
        #for i in range(8, 11 + 1):
        for i in range(8, 10 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = F.relu(self['conv{:d}_2'.format(i)](h))
            ys.append(h)
        return ys
        :
class SSD300(SSD):
            :
            multibox=Multibox(
                n_class=n_fg_class + 1,
                #aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))),
                aspect_ratios=((2, 3), (2, 3), (2, 3), (2,))),
            #steps=(8, 16, 32, 64, 100, 300),
            steps=(16, 32, 64, 100),
            #sizes=(30, 60, 111, 162, 213, 264, 315),
            sizes=(60, 111, 162, 213, 264),
            :
```
