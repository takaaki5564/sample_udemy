

# ChainerCV SSD on Google Colaboratory


### Check GPU
```
import tensorflow as tf
tf.test.gpu_device_name()
```

### Install Chainer and related libraries.

```
!curl https://colab.chainer.org/install | sh -
!ln -snf /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so.9.2 /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so
!pip install chainer
```

### Install ChainerCV and related libraries.

```
!pip install Pillow
!pip install Cython
!pip install Matplotlib
!pip install chainercv
```

### Get Image
```
!wget https:xxx.jpg
```
c.f. 
https://cdn.pixabay.com/photo/2017/04/27/11/21/dog-2265233_960_720.jpg

### Import Libraries

```
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv import utils
from chainercv.visualizations import vis_bbox
%matplotlib inline
```

### Execute Inference

```
model = SSD300(
        n_fg_class=len(voc_bbox_label_names),
        pretrained_model='voc0712')

chainer.cuda.get_device_from_id(0).use()
model.to_gpu()

img = utils.read_image('dog-2265233_960_720.jpg', color=True)
bboxes, labels, scores = model.predict([img])
bbox, label, score = bboxes[0], labels[0], scores[0]

vis_bbox(
    img, bbox, label, score, label_names=voc_bbox_label_names)
plt.show()
```

reference:
https://github.com/chainer/chainercv/blob/master/examples/ssd/demo.py


