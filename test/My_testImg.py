from newlandface import nlface,newlandface
import cv2

# obj = newlandface.analyze("000_0.bmp", actions = ['age', 'gender', 'race', 'emotion'])



nlface.load_model()
cv2.namedWindow("test",0)


image = cv2.imread("FaceDetection_test2.jpg")
faceObjs = nlface.detect_face(image,detector_backend = 'opencv')
# 第一版本，调用原始程序版本
# font = cv2.FONT_HERSHEY_SIMPLEX
# if faceObjs is not 0:
#     for idx, rect in enumerate(faceObjs):
#         # cv2.waitKey()
#         # 检测属性
#         actions = ['age','gender']
#         muilty = nlface.analyze(image, rect, actions=actions)
#         # 检测点
#         points = nlface.detect_points(image,rect)
#         # 显示画框
#         left = rect.left()
#         right = rect.right()
#         top = rect.top()
#         bottom = rect.bottom()		
#         cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),2)
#         cv2.imshow("test",image)
#         # 显示点
#         for i,point in enumerate(points):
#             cv2.circle(image,(point[0],point[1]),2,(0,0,255),-1)
#         # 显示属性
#         txtPrint = ''
#         for action in actions:
#             txtPrint = txtPrint+str(action)+ ":" + str(muilty[action]) + " "
#         cv2.putText(image, str(txtPrint), (left,int(bottom*1.2)), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
#         cv2.imshow("test",image)
#         cv2.waitKey()
#     cv2.waitKey()
# else:
#     print("no face detect")

# 第二版本，直接调用画框
if faceObjs is not 0:
    for idx, rect in enumerate(faceObjs):
        actions = ['age','gender']
        muilty = nlface.analyze(image, rect,actions=actions)
        image = nlface.show_face(image,rect,muilty,actions)
        # cv2.waitKey()
        # 检测点
        points = nlface.detect_points(image,rect)
        for i,point in enumerate(points):
            cv2.circle(image,(point[0],point[1]),2,(0,0,255),-1)
            cv2.imshow("test",image)
            cv2.waitKey(1)
        cv2.imshow("test",image)
        cv2.waitKey()
else:
    print("no face detect")



# 第三版本，直接调用画框、画点，如果人脸太多会很久才显示
# if faceObjs is not 0:
#     for idx, rect in enumerate(faceObjs):
#         image = nlface.show_face_points(image,rect)
#         cv2.imshow("test",image)
#         cv2.waitKey(1)
#     cv2.waitKey()
# else:
#     print("no face detect")

