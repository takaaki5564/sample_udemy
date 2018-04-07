# Training VOC dataset with TensorFlow Object Detection API on AWS

## Preparation

### Log in to your Deep Learning AMI (Ubuntu), instance type = p2.xlarge

Activate tensorflow with python3.6

```
source activate tensorflow_p36
```

### Setup Tensorflow

Download Tensorflow object detection model from git

```
cd ~
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git
```

Build model
```
cd models/research
protoc object_detection/protos/*.proto --python_out=.
python setup.py build
python setup.py install
```

Build slim model
```
cd models/research/slim
python setup.py build
python setup.py install
cd models/research
```

Set path
```
echo 'export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research:~/tensorflow/models/research/slim:~/tensorflow/models/research/object_detection' >> ~/.bashrc
```

Check build
'''
python object_detection/builders/model_builder_test.py
'''

### Prepare VOC dataset

Upgrade xml

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

### Edit configuration

Edit your own config file
```
cp ./samples/configs/ssd_inception_v3_pets.config ./ssd_inception_v3_voc.config
vim ./ssd_inception_v3_voc.config
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
  fine_tune_checkpoint: "./my_model/inception_v3.ckpt"
  from_detection_checkpoint: false
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
mkdir my_model
python ./train.py --logtostderr --train_dir=./my_model --pipeline_config_path=ssd_inception_v3_voc.config
```

## Monitoring progress on Tensorboard

Launch Tensorboard process
```
tensorboard --logdir=./my_model/
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

![screenshot](/images/tensorboard_train.png)

### Evaluate trained model on Tensorboard

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
mkdir my_eval
python eval.py --logtostderr --pipeline_config_path=ssd_inception_v3_voc.config --checkpoint_dir=my_model/ --eval_dir=my_eval/
```

Visualize eval results
```
tensorboard --logdir=my_eval/
```

Open the browser and type in address
```
http://YourInstancePublicDNS:6006
```

![screenshot](/images/tensorboard_eval.png)


## Retrain model

If you want to retrain the model, edit your own config file as follows.

```
vim ./ssd_inception_v3_voc.config
```

 * Comment out chekpoint for base model
 * Set from_detection_checkpoint as "true"
 
```
  #fine_tune_checkpoint: "./my_model/inception_v3.ckpt"
  from_detection_checkpoint: true
  :
  
```

Start training
```
python ./train.py --logtostderr --train_dir=./my_model --pipeline_config_path=ssd_inception_v3_voc.config
```
