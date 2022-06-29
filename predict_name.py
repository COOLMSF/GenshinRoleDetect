from distutils.log import info
from importlib import import_module
from statistics import mode
from unittest import result
from PIL import Image
from joblib import parallel_backend
import paddle
import paddle.fluid as fluid
import numpy as np
import sys
from multiprocessing import cpu_count
import time 
import matplotlib.pyplot as plt
from pyparsing import identbodychars
paddle.enable_static()

def load_img(path):
    img = paddle.dataset.image.load_and_transform(path,128,128,False).astype("float32")
    img = img / 255.0
    return img

place = fluid.CUDAPlace(0) # 指定CPU
infer_exe = fluid.Executor(place)# 指定场所
model_path = "model/" # 模型路径
# 加载模型
infer_prog,feed_names,fetch_targets = fluid.io.load_inference_model(model_path,infer_exe)

# 设置识别的图片
test_img = "2.jpg"
infer_imgs = []
infer_imgs.append(load_img(test_img))
infer_imgs = np.array(infer_imgs) 

# 执行预测
params = {feed_names[0]:infer_imgs}
result = infer_exe.run(infer_prog,feed=params,fetch_list=fetch_targets)

# 取出概率最高的模型
idx = np.argmax(result[0][0])

name_dict = {'keli':0,'hutao':1,'abeiduo':2}
# 将idx对应数字转换为字典中的字符串
for k,v in name_dict.items():
    if idx == v:
        print("预测结果:",k)

img = Image.open(test_img)
plt.imshow(img)
plt.show()
