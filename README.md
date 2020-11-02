---
typora-copy-images-to: upload
---

# newlandface
[TOC]

newlandface 是一个轻量级的人脸检测和多属性分析（年龄、性别、标签）分析工具，使用python语言，集成了部分开源库Dlib、mtcnn等工具，该库主要是基于Keras和TensorFlow来进行开发的。

## Installation 安装

The easiest way to install newlandface is to download it from [`PyPI`](https://pypi.org/project/newlandface/).

```python
pip install newlandface
```

## 文件结构
> │  README.md
> │  requirements.txt
> │  setup.py
> │  
> ├─newlandface
> │  │  nlface.py
> │  │  __init__.py
> │  │  
> │  ├─basemodels
> │  │  │  DeepID.py
> │  │  │  DlibResNet.py
> │  │  │  Facenet.py
> │  │  │  FbDeepFace.py
> │  │  │  OpenFace.py
> │  │  │  VGGFace.py
> │  │  │  __init__.py
> │  │          
> │  ├─commons
> │  │  │  distance.py
> │  │  │  functions.py
> │  │  │  realtime.py
> │  │  │  __init__.py
> │  │          
> │  ├─extendedmodels
> │  │  │  Age.py
> │  │  │  Emotion.py
> │  │  │  Gender.py
> │  │  │  Race.py
> │  │  │  __init__.py
> │  │          
> │  ├─models
> │  │      face-recognition-ensemble-model.txt
> │  │      __init__.py
> │      
> └─tests
>     │  testFaceAttr_video.py
>     │  testFaceDetect_img.py
>     │  testFacePoints_img.py
>     │  
>     └─dataset
>             img1.jpg
>             img13.jpg
>             img14.jpg
>             ...

## 测试代码

### 1.1 人脸检测代码

```python
from newlandface import nlface
import cv2
# 模型加载
nlface.load_model()
image = cv2.imread("./dataset/test1.jpg")
# 人脸检测
faceObjs = nlface.detect_face(image)
# 显示人脸框
if faceObjs is not 0:
    for idx, rect in enumerate(faceObjs):
        image = nlface.show_face(image,rect)
else:
    print("no face detect")
    os._exit(0)
cv2.imshow("test",image)
cv2.waitKey()
```

![facedetect](https://gitee.com/fjndfazp/picgo/raw/master/img/20201102193344.jpg)



### 1.2 人脸68点检测

#### 1.2.1 直接调用show_face_points函数进行显示

核心函数：***detect_face、show_face_points***

```python
from newlandface import nlface
import cv2
cv2.namedWindow("test",0)
# 模型加载
nlface.load_model()
image = cv2.imread("./dataset/test1.jpg")
# 人脸检测
faceObjs = nlface.detect_face(image)
# 显示人脸框
if faceObjs is not 0:
    for idx, rect in enumerate(faceObjs):
        image = nlface.show_face_points(image,rect)
else:
    print("no face detect")
    os._exit(0)
cv2.imshow("test",image)
cv2.waitKey()
```

#### 1.2.2 调用点检测模块，自行画图

核心函数：***detect_face、detect_points、show_face***

```python
from newlandface import nlface
import cv2
cv2.namedWindow("test",0)
# 模型加载
nlface.load_model()
image = cv2.imread("./dataset/test1.jpg")
# 人脸检测
faceObjs = nlface.detect_face(image)
if faceObjs is not 0:
    for idx, rect in enumerate(faceObjs):
        # 人脸68点检测
		points = nlface.detect_points(image,rect)
        # 显示人脸框、68点
		image = nlface.show_face(image,rect)
        for i,point in enumerate(points):
            cv2.circle(image,(point[0],point[1]),2,(0,0,255),-1)
            cv2.imshow("test",image)
        cv2.waitKey(1)    
else:
    print("no face detect")
    os._exit(0)
cv2.imshow("test",image)
cv2.waitKey()
```



![facepoints_test](https://gitee.com/fjndfazp/picgo/raw/master/img/20201102193336.jpg)



### 1.3 人脸属性分析

核心函数：***detect_face、detect_points、show_face***

属性开放：**emotion(表情)、age（年龄）、gender（性别）**

|    属性     | 检测耗时 |                           标签类型                           |
| :---------: | :------: | :----------------------------------------------------------: |
| emotion表情 |   30ms   | ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] |
|   age年龄   |  130ms   |                            1-100                             |
| gender性别  |  170ms   |                          woman、man                          |

注意：不同的模块耗时不一，所以如果调用摄像头的时候，要注意实时性上的要求。

```python
from newlandface import nlface
import cv2
cv2.namedWindow("test",0)
# 模型加载
nlface.load_model()
image = cv2.imread("./dataset/test1.jpg")
# 人脸检测
faceObjs = nlface.detect_face(image)
if faceObjs is not 0:
    for idx, rect in enumerate(faceObjs):
        # 人脸属性分析
        actions = ['emotion', 'age', 'gender']
        attribute = nlface.analyze(image, faceObjs[idex],actions = actions)
        # 显示人脸框\属性
		image = nlface.show_face(image,rect)
        image = nlface.show_face_attr(image, faceObjs[idex], attribute, actions)
        cv2.imshow("test",image)
        cv2.waitKey(1)    
else:
    print("no face detect")
    os._exit(0)
cv2.imshow("test",image)
cv2.waitKey()
```

![faceAttr](https://gitee.com/fjndfazp/picgo/raw/master/img/20201102193325.jpg)









