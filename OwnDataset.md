
### Train SSD with your own dataset

Use image-labelling-tool

https://github.com/takaaki5564/image-labelling-tool

<br>

Setup python 3.6 environment on your local PC and install image-labeling-tool.

```
git clone https://github.com/takaaki5564/image-labelling-tool.git
cd image-labeling-tool
pip install -e .
```

Setup your own dataset as follows.

1. Copy and rename example/simple directory (e.g. example/pet)
1. Edit label_name_example.yml and register your labels (e.g. dog and cat)
1. Collect images and put them into /images directory (e.g. example/pet/images)
The oxford-IIIT Pet Dataset was used in this demo. (http://www.robots.ox.ac.uk/~vgg/data/pets/)
1. Run flask_app.py and open browser on your local PC (http://127.0.0.1:5000)
```
python flask_app.py --image_dir examples/pet/images --label_names examples/pet/label_names_example.yml --file_ext jpg
```
1. Label your own dataset
  1. Click "Draw box" button
  1. Left-click to drow bounding box for your interesting region
  1. Select labels in "Labels" pulldown menu
  1. Drow region to all images

（注意！）
ここで "Labels" が全て登録されていることを再度確認してください。
登録されていないと Label が NULL として登録され、正常に学習できません。


1. zip dataset
```
cd example
zip -r pet pet
cd ~
scp -i (keyfilename).pem ./image-labeling-tool/examples/pet.zip ubuntu@52.40.30.xxx:/home/ubuntu/
```


Connect to AWS server on another terminal
```
ssh -i (keyfilename).pem ubuntu@52.40.30.xxx
source activate chainer
ls
```

Install image-labeling-tool on AWS server
```
git clone https://github.com/takaaki5564/image-labelling-tool.git
cd image-labeling-tool
pip install -e .
```

Unzip labeled images under /ssd directory
```
cd ~/image-labeling-tool/example
mv ~/pet.zip ./ssd/
unzip pet.zip
rm pet.zip
```

Run randomly_split_directory.py and divide your own dataset to train or val

https://github.com/takaaki5564/image-labelling-tool/tree/master/examples/ssd
```
python randomly_split_directory.py ./pet/train ./pet/val ./pet/images/
```

Start training with your own dataset
```
python train.py --train ./pet/train --val ./pet/val --label_names ./pet/label_names_example.yml --val_iteration 10  --gpu 0 --batchsize 4 --log_iteration 10 --lr 0.00004
```

Evaluate trained model
```
vim demo.py
```

Edit demo.py as follows

```
#plot.show()
save_image = args.image.replace(".jpg", "_out.png")
plot.savefig(save_image)
```

Run demo.py
```
python demo.py --pretrained_model result/model_iter_400 --label_names apple_orange_annotations/apple_orange_label_names.yml apple_orange_annotations/Orange/frame0017.jpg
```

Download data from server to local
```
scp -i (keyfilename).pem ubuntu@52.40.30.xxx:/home/ubuntu/image-labeling-tool/examples/ssd/pet/val/*.png .
```
