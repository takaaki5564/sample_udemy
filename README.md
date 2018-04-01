# Training your TensorFlow Object Detection API on AWS


### Log in to your Deep Learning AMI (Ubuntu) and activate tensorflow with python3.6

```
source activate tensorflow_p36
```

### Setup Tensorflow model

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

 
