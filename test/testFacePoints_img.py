from newlandface import nlface
import cv2

cv2.namedWindow("test",0)
image = cv2.imread("test2.jpg")
faceObjs = nlface.detect_face(image,detector_backend = 'opencv')
if faceObjs is not 0:
    for idx, rect in enumerate(faceObjs): 
        left = rect.left(); 
        right = rect.right()
        top = rect.top(); 
        bottom = rect.bottom()		
        cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),2)
        cv2.imshow("test",image)
        # cv2.waitKey()
        # 检测点
        points = nlface.detect_points(image,rect)
        for i,point in enumerate(points):
            cv2.circle(image,(point[0],point[1]),2,(0,0,255),-1)
            cv2.imshow("test",image)
            cv2.waitKey(1)
    cv2.waitKey()
else:
    print("no face detect")



# cv2.namedWindow("test",0)
# image = cv2.imread("test2.jpg")
# faceObjs = nlface.detect_face(image)
# for idx, rect in enumerate(faceObjs): 
#     left = rect[0]; 
#     right = rect[0]+rect[2]
#     top = rect[1] 
#     bottom = rect[1]+rect[3]	
#     cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),2)
#     cv2.imshow("test",image)
#     # cv2.waitKey()
#     # 检测点
#     points = nlface.detect_points(image,rect)
#     for i,point in enumerate(points):
#         cv2.circle(image,(point[0],point[1]),2,(0,0,255),-1)
#         cv2.imshow("test",image)
#         cv2.waitKey(1)
     
# cv2.waitKey()
# detected_face = img[top:bottom, left:right]	



# obj = nlface.analyze("000_0.bmp", actions = ['age', 'gender', 'race', 'emotion'])
# #faceObjs = nlface.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
# print(obj["age"]," years old ", obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])