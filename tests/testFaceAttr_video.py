from newlandface import nlface
import cv2
import time

print("开始加载模型")
start = time.time()
nlface.load_model()
end = time.time()
print("模型加载时间:%.2f秒"%(end-start))

camera = cv2.VideoCapture(0)
cv2.namedWindow('Camera',0)
while True:
    success, frame = camera.read()
    if frame is None:
        print("Capture image failed!")
        break
    start = time.time()
    faceObjs = nlface.detect_face(frame)
    end = time.time()
    print("检测运行时间:%.2f秒"%(end-start))
    if faceObjs is not 0:
        start = time.time()
        # 检测人脸属性
        actions = ['emotion', 'age', 'gender']
        attribute = nlface.analyze(frame, faceObjs[0],actions = actions)
        end = time.time()
        print("属性分析时间:%.2f秒"%(end-start))
        # 显示人脸框和属性
        frame = nlface.show_face_attr(frame, faceObjs[0], attribute, actions)
        # 检测点
        points = nlface.detect_points(frame,faceObjs[0])
        for i,point in enumerate(points):
            cv2.circle(frame,(point[0],point[1]),2,(0,0,255),-1)
    cv2.imshow("Camera",frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyWindow('Camera')
camera.release()



