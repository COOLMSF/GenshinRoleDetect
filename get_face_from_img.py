#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import base64
from nis import cat
import requests
import urllib
from PIL import Image
import os
import time
import sys

# 获取路径为img的人脸box
def get_face(img):
    try:
        request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect" #请求地址
        with open(img,"rb") as f:
            base64_data = base64.b64encode(f.read())
            params = {
                "image":str(base64_data,encoding='UTF8'),
                "image_type":"BASE64"
            }
            # urllib.p
            params = urllib.parse.urlencode(params).encode(encoding='UTF8')
            request_url = request_url + "?access_token=" + access_token
            request = urllib.request.Request(url=request_url, data=params)
            request.add_header('Content-Type', 'application/json')
            response = urllib.request.urlopen(request)
            content = response.read()
            content = json.loads(str(content,encoding='UTF8')) #解析请求结果
            # print(content)
            face_location = content['result']['face_list'][0]['location']
            return(face_location)
    except:
        print("从服务器获取数据出错")

# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=QHyjVtWGmG9rgRiCtIML5zjv&client_secret=sRgITz59y61SRzePP1BHn0koaFQFCLF9'
response = requests.get(host)
if response:
    content = response.json()
    access_token = content['access_token']
print(access_token)

img = "keli.jpeg"

def get_save_img_face(src, dst):
    name = src.split('/')[-1]

    face_location = get_face(src)
    # (left, upper, right, lower)
    try:
        box = (face_location['left']-50,face_location['top']-150,face_location['left']+face_location['width']+50,face_location['top']+face_location['height']+50)
        img = Image.open(src)
        img = img.crop(box)
        img.save(dst + name) #保存裁剪完的人脸图片
    except:
        print("定位出错")


if __name__ == '__main__':
    print(os.getcwd())

    if not os.path.exists("result"):
        os.mkdir("result")
    for img in os.listdir():
        if "jpeg" or "png" or "jpg" in img:
            get_save_img_face(img, "result/")
            time.sleep(2)
