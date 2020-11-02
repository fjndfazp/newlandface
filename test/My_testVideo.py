from newlandface import nlface
import cv2
import time

# camera = cv2.VideoCapture(0)
# cv2.namedWindow('MyCamera',0)
# while True:
#     success, frame = camera.read()
#     start = time.time()
#     faceObjs = nlface.detect_face(frame,detector_backend = 'opencv')
#     end = time.time()
#     print("循环运行时间:%.2f秒"%(end-start))
#     if faceObjs is not 0:
#         frame = nlface.show_face(frame,faceObjs[0])
#         start = time.time()
#         points = nlface.detect_points(frame,faceObjs[0])
#         end = time.time()
#         print("点检测时间:%.2f秒"%(end-start))
#         for i,point in enumerate(points):
#             cv2.circle(frame,(point[0],point[1]),2,(0,0,255),-1)
#             # cv2.imshow("test",frame)
#             # cv2.waitKey(1)
#     cv2.imshow("MyCamera",frame)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
# cv2.destroyWindow('MyCamera')
# camera.release()



