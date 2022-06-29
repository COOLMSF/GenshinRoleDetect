import multiprocessing
from operator import imod
import os
import random
from statistics import mode
import paddle
import paddle.fluid as fluid
import numpy as np
import sys
from multiprocessing import cpu_count
import time 
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image

paddle.enable_static()

fruits_path = './data/faces'
fruits_dirs = os.listdir(fruits_path)
fruits_dict = {}

def add_to_dict(path,name):
    if name not in fruits_dict:
        img_list = []
        img_list.append(path)
        fruits_dict[name] = img_list
    else:
        fruits_dict[name].append(path)

for i in fruits_dirs:
    full_path = os.path.join(fruits_path,i)
    imgs = os.listdir(full_path)
    for img in imgs:
        img_full_path =full_path +  "/" + img 
        add_to_dict(img_full_path,i)


train_path_file = "train.txt"
test_path_file = "test.txt"
name_dict = {'keli':0,'hutao':1,'abeiduo':2}

for name,img_list in fruits_dict.items():
    i = 0
    num = len(img_list)
    print("%s:%d" % (name,num))
    for im in img_list:
        line = "%s\t%d\n" % (im,name_dict[name])
        if (i % 10 == 0):
            with open(test_path_file,'a') as f:
                f.write(line)
        else:
            with open(train_path_file,'a') as f:
                f.write(line)
        i += 1

def train_mapper(sample):
    '''
    param sample (路径，类型)
    return 图像的张量
    '''
    #img 图片路径,label类别
    img,label = sample
    #读取图片
    img = paddle.dataset.image.load_image(img)
    # 标准化 
    img = paddle.dataset.image.simple_transform(im = img,crop_size = 128,resize_size=128,is_color=True, is_train=True)
    # 归一化,统一转为百分数
    img = img.astype("float32") / 255.0
    return img,label

# 训练集读取样本 
'''
    param train_list 训练集路径
    param buffer_size 读取数量

'''
def train_r(train_list,buffer_size=1024):
    def reader():
        with open (train_list,"r") as f:
            for line in f.readlines():
                line = line.replace("\n","")
                img_path,lbl = line.split("\t")
                yield img_path,int(lbl)
    return paddle.reader.xmap_readers(train_mapper,reader,cpu_count(),buffer_size) 


''' param image 归一化与标准化的图像数据
    param type_size 5 
    return predict 
'''
def Create_CNN(image,type_size):
    # conv_pool_1
    # 卷积池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input = image,num_filters=32,filter_size=3,pool_size=2,pool_stride=2,act="relu")
    # 丢弃数据，防止过拟合
    drop = fluid.layers.dropout(x=conv_pool_1,dropout_prob=0.5) # 丢弃率 

    # conv_pool_2
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input = image,num_filters=64,filter_size=3,pool_size=2,pool_stride=2,act="relu")
    drop = fluid.layers.dropout(x=conv_pool_2,dropout_prob=0.5) # 丢弃率 

    # conv_pool_3
    conv_pool_3 = fluid.nets.simple_img_conv_pool(input = image,num_filters=64,filter_size=3,pool_size=2,pool_stride=2,act="relu")
    drop = fluid.layers.dropout(x=conv_pool_3,dropout_prob=0.5) # 丢弃率 

    # fc1  
    # 全连接层
    fc = fluid.layers.fc(input=drop,size=512,act="relu")
    drop = fluid.layers.dropout(x=fc,dropout_prob=0.5) # 丢弃率 

    # output
    predict = fluid.layers.fc(input=drop,size=type_size,act="softmax")
    return predict

# 批次大小
batch_size = 32
#读取数据到train_reader
train_reader = train_r(train_path_file)
# 打乱数据
random_train_reader = paddle.reader.shuffle(train_reader,buf_size=1300)
# 批量读取
batch_train_reader = paddle.batch(random_train_reader,batch_size=batch_size)

# 占位符
image = fluid.layers.data(name='image',shape=[3,128,128],dtype="float32")
label = fluid.layers.data(name='label',shape=[1],dtype="int64")


# 调用函数
predict = Create_CNN(image,3)
cost = fluid.layers.cross_entropy(input=predict,label=label)

#损失函数均值化
avg_cost  = fluid.layers.mean(cost) 

#优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.01)
optimizer.minimize(avg_cost)

#计算准确率
accuracy =  fluid.layers.accuracy(input=predict,label=label)
#执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

#喂入数据
feeder = fluid.DataFeeder(place=place,feed_list=[image,label])

# 开始训练
for epoch in range(50):
    for bat_id,data in enumerate(batch_train_reader()):
        train_cost,train_acc = exe.run(fluid.default_main_program(),feed=feeder.feed(data),fetch_list=[avg_cost,accuracy])
        if bat_id % 20 == 0:
            print("epoch:%d,batch:%d,cost:%f,acc:%f" % (epoch,bat_id,train_cost[0],train_acc[0]))


model_dir = "model/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
fluid.io.save_inference_model(model_dir,['image'],[predict],exe)
