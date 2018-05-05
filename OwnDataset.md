
## Train SSD with your own dataset

<br>

Use image-labelling-tool which was modified for ChainerCV SSD

https://github.com/takaaki5564/image-labelling-tool

https://github.com/takaaki5564/image-labelling-tool/tree/master/examples/ssd


<br>

### Preparation

Setup python 3.6 environment on your local PC and install image-labeling-tool.

```
git clone https://github.com/takaaki5564/image-labelling-tool.git
cd image-labeling-tool
pip install -e .
```

<br>

### Label Images

Collect images

1. Copy and rename example/simple directory (e.g. example/pet)
1. Edit label_name_example.yml and specify labels (e.g. dog and cat)
1. Collect images into /images directory (e.g. example/pet/images)

The oxford-IIIT Pet Dataset was used in this demo. (http://www.robots.ox.ac.uk/~vgg/data/pets/)

<br>

Launch image-labeling-tool

1. Run flask_app.py
```
python flask_app.py --image_dir examples/pet/images --label_names examples/pet/label_names_example.yml --file_ext jpg
```
1. Open web browser on your local PC and access to http://127.0.0.1:5000
1. Label images
    1. Click "Draw box" button
    1. Draw bounding box for the labeled region using left-click
    1. Select labels in "Labels" pulldown menu (\*1)
    1. Draw bounding box for all the images

\*1
最後に全ての画像に "Labels" が登録されていることを確認してください。
登録されない画像がある場合、ラベルが NULL と登録されて正常に学習が開始されません。


Zip labeld dataset and Upload to server using scp

```
cd example
zip -r pet pet
cd ~
scp -i (keyfilename).pem ./image-labeling-tool/examples/pet.zip ubuntu@52.40.30.xxx:/home/ubuntu/
```

<br>

### Train SSD-300 with your own Dataset

Launch another terminal and connect to server

```
ssh -i (keyfilename).pem ubuntu@52.40.30.xxx
source activate chainer
ls
```

Install image-labeling-tool on server

```
git clone https://github.com/takaaki5564/image-labelling-tool.git
cd image-labeling-tool
pip install -e .
```

Unzip labeled images under ~/image-labeling-tool/example/ssd directory

```
cd ~/image-labeling-tool/example
mv ~/pet.zip ./ssd/
unzip pet.zip
rm pet.zip
```

Divide dataset to train/val

```
python randomly_split_directory.py ./pet/train ./pet/val ./pet/images/
```

Start training with your own dataset

```
python train.py --train ./pet/train --val ./pet/val --label_names ./pet/label_names_example.yml --val_iteration 10  --gpu 0 --batchsize 4 --log_iteration 10 --lr 0.00004
```

<br>

### Evaluate Train Result

```
vim demo.py
```

Modify demo.py to save result images as follows

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
