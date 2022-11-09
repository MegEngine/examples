本目录包含了采用MegEngine实现的经典`ResNet`网络结构，同时提供了在ImageNet训练集上的完整训练代码，以及int4模型转换教程。

## 数据集准备
创建data文件夹
在文件夹中下载`ILSVRC2012_devkit_t12.tar.gz`  `ILSVRC2012_img_train.tar`  `ILSVRC2012_img_val.tar`并解压

## float模型训练
注意：-n 8 表示8卡训练。需要根据实际卡的数量进行调整。
```
python3 train.py -d data/ -a resnet50 -n 8 --save log --epochs 90 -b 64 --lr 0.025 --momentum 0.9 --weight-decay 1e-4 --mode normal -j 4
```

## float int4校准
```
python3 calibrate.py -d data/ -a resnet50 -m log/resnet50/normal/checkpoint.pkl -o model_qat_init.pkl
```

## qat模型训练
```
python3 train.py -d data/ -a resnet50 -n 8 --save log --epochs 10 -b 64 --lr 0.00025 --momentum 0.9 --weight-decay 1e-5 --mode qat -j 4 -m model_qat_init.pkl --warmup_epochs 0
```

## 模型 dump 成静态图 .mge 格式 
```
python3 dump.py -m log/resnet50/qat/checkpoint.pkl
ls resnet50-int4.mge
``` 

