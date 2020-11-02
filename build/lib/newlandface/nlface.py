from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
import time
import os
from os import path
from pathlib import Path
import gdown
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import cv2
from keras import backend as K
import keras
import tensorflow as tf
import pickle

from newlandface import nlface
from newlandface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID
from newlandface.extendedmodels import Age, Gender, Race, Emotion
from newlandface.commons import functions, realtime, distance as dst


from newlandface.commons import distance
import math
from PIL import Image
import dlib 

home = 'D:/newlandface'
predictor = ''
detector = ''
ssd_detector = ''
emotion_model = ''
age_model = ''
gender_model = ''
race_model = ''

def load_model(actions = []):
	global home
	global predictor
	global detector
	global ssd_detector
	global emotion_model
	global age_model
	global gender_model
	global race_model
	predictor = dlib.shape_predictor(home+"/weights/shape_predictor_68_face_landmarks.dat")
	detector = dlib.get_frontal_face_detector()
	ssd_detector = cv2.dnn.readNetFromCaffe(
		home+"/weights/deploy.prototxt", 
		home+"/weights/res10_300x300_ssd_iter_140000.caffemodel"
	)
    #if a specific target is not passed, then find them all
	if len(actions) == 0:
		actions= ['emotion', 'age', 'gender'] #  'race'
	print("Actions to do: ", actions)
	if 'emotion' in actions:
		emotion_model = Emotion.loadModel()
	
	if 'age' in actions:
		age_model = Age.loadModel()
	
	if 'gender' in actions:
		gender_model = Gender.loadModel()

	if 'race' in actions:
		race_model = Race.loadModel()



def shape_to_np(shape, dtype='int'):
    # 创建68*2用于存放坐标
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # 遍历每一个关键点
    # 得到坐标
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def get_opencv_path():
	opencv_home = cv2.__file__
	folders = opencv_home.split(os.path.sep)[0:-1]

	path = folders[0]
	for folder in folders[1:]:
		path = path + "/" + folder

	return path+"/data/"

def detect_face(image,detector_backend = 'ssd'):
	global detector
	global ssd_detector	
	if detector_backend == 'ssd':
		ssd_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
		target_size = (300, 300)
		original_size = image.shape
		img = cv2.resize(image, target_size)
		aspect_ratio_x = (original_size[1] / target_size[1])
		aspect_ratio_y = (original_size[0] / target_size[0])
		imageBlob = cv2.dnn.blobFromImage(image = img)
		ssd_detector.setInput(imageBlob)
		detections = ssd_detector.forward()
		detections_df = pd.DataFrame(detections[0][0], columns = ssd_labels)
		detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
		detections_df = detections_df[detections_df['confidence'] >= 0.80]
		detections_df['left'] = (detections_df['left'] * 300*aspect_ratio_x).astype(int)
		detections_df['bottom'] = (detections_df['bottom'] * 300*aspect_ratio_y).astype(int)
		detections_df['right'] = (detections_df['right'] * 300*aspect_ratio_x).astype(int)
		detections_df['top'] = (detections_df['top'] * 300*aspect_ratio_y).astype(int)
		rects = []
		for index, facerect in detections_df.iterrows():
			rect = dlib.rectangle(int(facerect['left']),int(facerect['top'])\
				 ,int(facerect['right'] ),int(facerect['bottom']))
			rects.append(rect)

	if detector_backend == 'opencv' :
		opencv_path = get_opencv_path()
		face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
		if os.path.isfile(face_detector_path) != True:
			raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")
		face_detector = cv2.CascadeClassifier(face_detector_path)
		facerects = []
		rects=[]
		try: 
			facerects = face_detector.detectMultiScale(image, 1.3, 5)
			for i,facerect in enumerate(facerects):
				rect = dlib.rectangle(facerect[0],facerect[1] ,facerect[0]+facerect[2],facerect[1]+facerect[3])
				rects.append(rect)
		except:
			pass

	if detector_backend == 'dlib' :
		img = image[:, :, ::-1]
		#this is not a must library within newlandface. that's why, I didn't put this import to a global level. version: 19.20.0	
		rects = detector(img, 1)

	if len(rects) > 0:
		print(type(rects))
		return rects	
	else: #if no face detected
		return 0

def detect_points(image, rect):	
	global predictor
	img = image[:, :, ::-1]
	# Rect = dlib.rectangle(rect[0],rect[1] ,rect[0]+rect[2],rect[1]+rect[3])
	img_shape = predictor(img, rect)
	shape = shape_to_np(img_shape)
	if len(shape) > 0:
    		return shape	
	else: #if no face detected
		raise ValueError("Face could not be detected.\
		 Please confirm that the picture is a face photo or consider to set enforce_detection param to False.") 

	return 0


def show_face(image, rect):
    # 显示框
	# param image:输入图片,rgb类型
    # param rects： 检测到人脸的框数组（left,top,right,bottom)
	# return ：返回绘制后的图片，用于显示结果
	font = cv2.FONT_HERSHEY_SIMPLEX
	left = rect.left()
	right = rect.right()
	top = rect.top(); 
	bottom = rect.bottom()		
	cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),2)
	return image

def show_face_points(image,rect):
    # 显示框和点
	# param image:输入图片,rgb类型
    # param rects： 检测到人脸的框数组（left,top,right,bottom)
	# return ：返回绘制后的图片，用于显示结果
	left = rect.left()
	right = rect.right()
	top = rect.top(); 
	bottom = rect.bottom()		
	cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),2)
	points = detect_points(image,rect)
	for i,point in enumerate(points):
		cv2.circle(image,(point[0],point[1]),2,(0,0,255),-1)
	return image


def show_face_attr(image, rect, attribute=None, actions=[]):
    # 显示框和属性
	# param image:输入图片,rgb类型
    # param rects： 检测到人脸的框数组（left,top,right,bottom)
	# return ：返回绘制后的图片，用于显示结果
	font = cv2.FONT_HERSHEY_SIMPLEX
	left = rect.left()
	right = rect.right()
	top = rect.top(); 
	bottom = rect.bottom()		
	cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),2)
	if attribute is not None and len(actions) !=0:
		# 利用cv2.putText输出1-68
		txtPrint = ''
		for action in actions:
			txtPrint = txtPrint + str(action)+ ":" + str(attribute[action]) + " "
		cv2.putText(image, str(txtPrint), (left,bottom), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)	
	return image



def alignment_procedure(img, left_eye, right_eye):
		
	#this function aligns given face in img based on left and right eye coordinates
	
	left_eye_x, left_eye_y = left_eye
	right_eye_x, right_eye_y = right_eye
	
	#-----------------------
	#find rotation direction
		
	if left_eye_y > right_eye_y:
		point_3rd = (right_eye_x, left_eye_y)
		direction = -1 #rotate same direction to clock
	else:
		point_3rd = (left_eye_x, right_eye_y)
		direction = 1 #rotate inverse direction of clock
	
	#-----------------------
	#find length of triangle edges
	a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
	b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
	c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))
	
	#-----------------------
	
	#apply cosine rule
			
	if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
		
		cos_a = (b*b + c*c - a*a)/(2*b*c)
		angle = np.arccos(cos_a) #angle in radian
		angle = (angle * 180) / math.pi #radian to degree
		
		#-----------------------
		#rotate base image
		
		if direction == -1:
			angle = 90 - angle
		
		img = Image.fromarray(img)
		img = np.array(img.rotate(direction * angle))
	
	#-----------------------
	
	return img #return img anyway
	
	
def align_face(img, rect, grayscale = False,target_size = (224, 224), detector_backend = 'opencv'):
	cropImg = img[rect.top():rect.bottom(), rect.left():rect.right()]	
	if (detector_backend == 'opencv') or (detector_backend == 'ssd'):
		opencv_path = get_opencv_path()
		eye_detector_path = opencv_path + "haarcascade_eye.xml"
		eye_detector = cv2.CascadeClassifier(eye_detector_path)
		# cv2.imshow("test11",cropImg)
		# cv2.waitKey()
		detected_face_gray = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY) #eye detector expects gray scale image
		eyes = eye_detector.detectMultiScale(detected_face_gray)
		if len(eyes) >= 2:
			#find the largest 2 eye
			base_eyes = eyes[:, 2]
			items = []
			for i in range(0, len(base_eyes)):
				item = (base_eyes[i], i)
				items.append(item)
			df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)
			eyes = eyes[df.idx.values[0:2]] #eyes variable stores the largest 2 eye
			#-----------------------
			#decide left and right eye
			eye_1 = eyes[0]; eye_2 = eyes[1]
			if eye_1[0] < eye_2[0]:
				left_eye = eye_1; right_eye = eye_2
			else:
				left_eye = eye_2; right_eye = eye_1
			#-----------------------
			#find center of eyes
			left_eye = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
			right_eye = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
			cropImg = alignment_procedure(cropImg, left_eye, right_eye)

		if grayscale == True:
			cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY) # opencv 的话就是rgb，如果是自带的bgr
		cropImg = cv2.resize(cropImg,  target_size)
		img_pixels = image.img_to_array(cropImg)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		img_pixels /= 255 #normalize input in [0, 1]
		return img_pixels #return img anyway	


def analyze(img, rect, actions = []):
	if len(actions) == 0:
		actions= ['emotion', 'age', 'gender']
	#--------------------------------
	disable_option = False if len(actions) > 1 else True
	pbar = tqdm(range(0,len(actions)), desc='Finding actions', disable = disable_option)
	action_idx = 0
	img_224 = None # Set to prevent re-detection
	resp_obj = "{"
	for index in pbar:
		action = actions[index]
		pbar.set_description("Action: %s" % (action))
		if action_idx > 0:
			resp_obj += ", "
		if action == 'emotion':
			emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
			# emotion_labels = ['生气', '厌恶', '害怕', '开心', '悲伤', '惊讶', '无表情']
			imgcrop = align_face(img, rect, grayscale = True,target_size = (48, 48))
			emotion_predictions = emotion_model.predict(imgcrop)[0,:]
			sum_of_predictions = emotion_predictions.sum()
			emotion_obj = "\"emotion\": {"
			for i in range(0, len(emotion_labels)):
				emotion_label = emotion_labels[i]
				emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
				if i > 0: emotion_obj += ", "
				emotion_obj += "\"%s\": %s" % (emotion_label, emotion_prediction)
			emotion_obj += "}"
			emotion_obj += ", \"emotion\": \"%s\"" % (emotion_labels[np.argmax(emotion_predictions)])
			resp_obj += emotion_obj

		elif action == 'age':
			if img_224 is None:
				img_224 = align_face(img, rect, grayscale = False, target_size = (224, 224))
			#print("age prediction")
			age_predictions = age_model.predict(img_224)[0,:]
			apparent_age = int(Age.findApparentAge(age_predictions))
			resp_obj += "\"age\": %s" % (apparent_age)

		elif action == 'gender':
			if img_224 is None:
				img_224 = align_face(img, rect,grayscale = False, target_size = (224, 224))
			#print("gender prediction")
			gender_prediction = gender_model.predict(img_224)[0,:]
			if np.argmax(gender_prediction) == 0:
				gender = "Woman"
			elif np.argmax(gender_prediction) == 1:
				gender = "Man"

			resp_obj += "\"gender\": \"%s\"" % (gender)

		elif action == 'race':
			if img_224 is None:
				img_224 = align_face(img, rect,grayscale = False, target_size = (224, 224))
			race_predictions = race_model.predict(img_224)[0,:]
			race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
			sum_of_predictions = race_predictions.sum()
			race_obj = "\"race\": {"
			for i in range(0, len(race_labels)):
				race_label = race_labels[i]
				race_prediction = 100 * race_predictions[i] / sum_of_predictions
				if i > 0: race_obj += ", "
				race_obj += "\"%s\": %s" % (race_label, race_prediction)
			race_obj += "}"
			race_obj += ", \"dominant_race\": \"%s\"" % (race_labels[np.argmax(race_predictions)])
			resp_obj += race_obj
		action_idx = action_idx + 1
	resp_obj += "}"
	resp_obj = json.loads(resp_obj)
	return resp_obj





def allocateMemory():
	print("Analyzing your system...")
	functions.allocateMemory()

#---------------------------
#main

functions.initializeFolder()

