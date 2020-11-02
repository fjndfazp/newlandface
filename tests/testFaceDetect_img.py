from newlandface import nlface
import cv2
import os 
import time
path = os.path.dirname(__file__)

print("开始加载模型")
start = time.time()
nlface.load_model()
end = time.time()
print("模型加载时间:%.2f秒"%(end-start))


cv2.namedWindow("test",0)
image = cv2.imread(path + "./dataset/test1.jpg")
if image is None:
    print("no image!")
    os._exit(0)

# 先进行人脸检测    
faceObjs = nlface.detect_face(image)

## -------------------------------------------------------------- # 
# 第1版本，直接调用画框、画点，如果人脸太多会很久才显示
'''
功能：人脸检测、68点检测
'''
if faceObjs is not 0:
    for idx, rect in enumerate(faceObjs):
        image = nlface.show_face(image,rect) # 只做画框
        # image = nlface.show_face_points(image,rect) # 画框和点
else:
    print("no face detect")
    os._exit(0)
cv2.imshow("test",image)
cv2.waitKey()

## -------------------------------------------------------------- # 
## example2
# '''
# 功能：人脸检测、68点检测、人脸属性分析
# '''
# if faceObjs is not 0:
#     for idx, rect in enumerate(faceObjs):
#         # 检测属性
#         actions = ['age', 'gender'] #'emotion' 
#         attribute = nlface.analyze(image, rect,actions=actions)
#         # 检测68点
#         points = nlface.detect_points(image,rect)
#         # 显示框\点\属性
#         image = nlface.show_face_attr(image, rect, attribute, actions)
#         for i,point in enumerate(points):
#             cv2.circle(image,(point[0],point[1]),2,(0,0,255),-1)
# else:
#     print("no face detect")
#     os._exit(0)
# cv2.imshow("test",image)
# cv2.waitKey()


## -------------------------------------------------------------- # 
## example3
# '''
# 功能：人脸检测、68点检测、人脸属性分析
# '''
font = cv2.FONT_HERSHEY_SIMPLEX
if faceObjs is not 0:
    for idx, rect in enumerate(faceObjs):
        # 检测属性
        actions = ['age','gender']
        attribute = nlface.analyze(image, rect, actions=actions)
        # 检测点
        points = nlface.detect_points(image,rect)
        # 画框\点\属性	
        cv2.rectangle(image,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(255,0,0),2)
        for i,point in enumerate(points):
            cv2.circle(image,(point[0],point[1]),2,(0,0,255),-1)
        txtShow = ''
        for action in actions:
            txtShow = txtShow + str(action) + ":" + str(attribute[action]) + " "
        cv2.putText(image, str(txtShow), (rect.left(),int(rect.bottom()*1.2)), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
else:
    print("no face detect")
    os._exit(0)
cv2.imshow("test",image)
cv2.waitKey()


