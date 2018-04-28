

### ChainerCV SSD on Google Colaboratory

Install Chainer and related libraries.

```
!apt -y install libcusparse8.0 libnvrtc8.0 libnvtoolsext1
!ln -snf /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so.8.0 /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so
!pip install cupy-cuda80 chainer
```

Install ChainerCV and related libraries.

```
!pip install Pillow
!pip install Cython
!pip install Matplotlib
!pip install -U numpy
!pip install chainercv
```

Get Image
```
!wget https:xxx.jpg
```

Import Libraries

```
import argparse
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import utils
from chainercv.visualizations import vis_bbox
%matplotlib inline
```

Execute Inference

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

