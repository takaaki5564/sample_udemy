# Training with TensorFlow Object Detection API on AWS


### Log in to your Deep Learning AMI (Ubuntu)

activate tensorflow with python3.6

```
source activate tensorflow_p36
```

### Setup Tensorflow Model

git clone Tensorflow model

```
cd ~
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git
```

build model
```
cd models/research
protoc object_detection/protos/*.proto --python_out=.
python setup.py build
python setup.py install
```

build slim model
```
cd models/research/slim
python setup.py build
python setup.py install
cd models/research
```

set path
```
echo 'export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research:~/tensorflow/models/research/slim:~/tensorflow/models/research/object_detection' >> ~/.bashrc
```

check build
'''
python object_detection/builders/model_builder_test.py
'''

### Prepare Dataset (VOC)

upgrade xml

```
pip install --upgrade lxml
```

Download VOC Dataset to your own directory
```
cd ~/tensorflow/models/research/object_detection
mkdir ./dataset
cd ./dataset
mkdir VOCtest
mkdir VOCtrain
cd VOCtest
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
cd ../VOCtrain
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
cd ~/tensorflow/models/research/object_detection
mkdir data/pascal_voc2007test
mkdir data/pascal_voc2007-2012train
```

Serialize xml for training data
```
python ./dataset_tools/create_pascal_tf_record.py --data_dir=./dataset/VOCtest/VOCdevkit ¥
 --year=VOC2007 ¥
 --set=test ¥
 --output_path=./data/pascal_voc2007test/pascal.record
python ./dataset_tools/create_pascal_tf_record.py --data_dir=./dataset/VOCtrain/VOCdevkit ¥
 --year=merged ¥
 --output_path=./data/pascal_voc2007-2012train/pascal.record
```

Edit your own config file
```
cp ./samples/configs/ssd_mobilenet_v1_pets.config ./ssd_mobilenet_v1_voc.config
vim ./ssd_mobilenet_v1_voc.config
```

* Change num_class
* Comment out file_tune_checkpoint
* Configure your own training data
* Configure your own evaluation data
```
model {
  ssd {
    num_classes: 21
    box_coder {
    :
    
  }
  #fine_tune_checkpoint: "./model/model.ckpt"
  from_detection_checkpoint: true
  :
  
train_input_reader: {
  tf_record_input_reader {
    input_path: "./data/pascal_voc2007-2012train/pascal.record"
  }
  label_map_path: "./data/pascal_label_map.pbtxt"
}
:

eval_input_reader: {
  tf_record_input_reader {
    input_path: "./data/pascal_voc2007test/pascal.record"
  }
  label_map_path: "./data/pascal_label_map.pbtxt"
  shuffle: false
  num_readers: 1
}
```

### Start training
```
python ./train.py --logtostderr --train_dir=./models --pipeline_config_path=ssd_mobilenet_v1_voc.config
```

### Monitoring training progress on Tensorboard

Launch Tensorboard process
```
tensorboard --logdir=./models/
```

Edit "Network & Security" on your EC2 dashboard.
```
Type : Custom TCP Rule
Protocol: TCP
Port Range: 6006
Source: Anywhere (0.0.0.0/0,::/0)
```

Open the browser and type in address
```
http://YourInstancePublicDNS:6006
```

![screenshot](/images/screenshot.png)

### Evaluate trained model

Install pycocotool
```
sudo pip install Cython
git clone https://github.com/cocodataset/cocoapi
cd PythonAPI
python setup.py build
python setup.py install
```

Evaluate trained model
```
python eval.py --logtostderr --pipeline_config_path=ssd_mobilenet_v1_voc.config --checkpoint_dir=models/ --eval_dir=eval
```

Visualize eval results
```
tensorboard --logdir=eval/
```
