from newlandface import nlface
import cv2
import time

nlface.load_model()
camera = cv2.VideoCapture(0)
cv2.namedWindow('MyCamera',0)
while True:
    success, frame = camera.read()
    start = time.time()
    faceObjs = nlface.detect_face(frame, detector_backend = 'opencv')
    end = time.time()
    print("检测运行时间:%.2f秒"%(end-start))
    if faceObjs is not 0:
        actions = ['gender']
        start = time.time()
        muilty = nlface.analyze(frame, faceObjs[0],actions = actions)
        print(muilty['gender'])
        end = time.time()
        print("属性分析时间:%.2f秒"%(end-start))
        frame = nlface.show_face(frame,faceObjs[0],muilty,actions)
        
        points = nlface.detect_points(frame,faceObjs[0])

        for i,point in enumerate(points):
            cv2.circle(frame,(point[0],point[1]),2,(0,0,255),-1)
            # cv2.imshow("test",frame)
            # cv2.waitKey(1)
    cv2.imshow("MyCamera",frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyWindow('MyCamera')
camera.release()




# obj = nlface.analyze("000_0.bmp", actions = ['age', 'gender', 'race', 'emotion'])
# #faceObjs = nlface.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
# print(obj["age"]," years old ", obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])