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

from newlandface import newlandface
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
		base_img = image.copy() #we will restore base_img to img later
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


def show_face(image,rect,muilty=None,actions=[]):
    # 显示
	# param image:输入图片,rgb类型
    # param rects： 检测到人脸的框数组（left,top,right,bottom)
	# param FaceNums: 最多显示人脸个数
	# param isPoints: 是否打印点的信息
	# return ：返回绘制后的图片，用于显示结果
	font = cv2.FONT_HERSHEY_SIMPLEX
	left = rect.left()
	right = rect.right()
	top = rect.top(); 
	bottom = rect.bottom()		
	cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),2)
	if muilty is not None and len(actions) !=0:
		# 利用cv2.putText输出1-68
		txtPrint = ''
		for action in actions:
			txtPrint = txtPrint + str(action)+ ":" + str(muilty[action]) + " "
		cv2.putText(image, str(txtPrint), (left,bottom), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)	
	return image

def show_face_points(image,rect):
    # 显示
	# param image:输入图片,rgb类型
    # param rects： 检测到人脸的框数组（left,top,right,bottom)
	# param FaceNums: 最多显示人脸个数
	# param isPoints: 是否打印点的信息
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
     
def detect_face_points(image,faceNum = 1):
	import dlib 
	img = image[:, :, ::-1]
	#this is not a must library within newlandface. that's why, I didn't put this import to a global level. version: 19.20.0
	detector = dlib.get_frontal_face_detector()
	rects = detector(img, 1)

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
		actions= ['emotion', 'age', 'gender', 'race']
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












def verify(img1_path, img2_path = '', model_name ='VGG-Face', distance_metric = 'cosine', model = None, enforce_detection = True, detector_backend = 'opencv'):
    
	tic = time.time()

	if type(img1_path) == list:
		bulkProcess = True
		img_list = img1_path.copy()
	else:
		bulkProcess = False
		img_list = [[img1_path, img2_path]]

	#------------------------------
	
	resp_objects = []
	
	if model_name == 'Ensemble':
		print("Ensemble learning enabled")
		
		import lightgbm as lgb #lightgbm==2.3.1
		
		if model == None:
			model = {}
			
			model_pbar = tqdm(range(0, 4), desc='Face recognition models')
			
			for index in model_pbar:
				
				if index == 0:
					model_pbar.set_description("Loading VGG-Face")
					model["VGG-Face"] = VGGFace.loadModel()
				elif index == 1:
					model_pbar.set_description("Loading Google FaceNet")
					model["Facenet"] = Facenet.loadModel()
				elif index == 2:
					model_pbar.set_description("Loading OpenFace")
					model["OpenFace"] = OpenFace.loadModel()
				elif index == 3:
					model_pbar.set_description("Loading Facebook newlandface")
					model["newlandface"] = FbDeepFace.loadModel()
					
		#--------------------------
		#validate model dictionary because it might be passed from input as pre-trained
		
		found_models = []
		for key, value in model.items():
			found_models.append(key)
		
		if ('VGG-Face' in found_models) and ('Facenet' in found_models) and ('OpenFace' in found_models) and ('newlandface' in found_models):
			print("Ensemble learning will be applied for ", found_models," models")
		else:
			raise ValueError("You would like to apply ensemble learning and pass pre-built models but models must contain [VGG-Face, Facenet, OpenFace, newlandface] but you passed "+found_models)
			
		#--------------------------
		
		model_names = ["VGG-Face", "Facenet", "OpenFace", "newlandface"]
		metrics = ["cosine", "euclidean", "euclidean_l2"]
		
		pbar = tqdm(range(0,len(img_list)), desc='Verification')
		
		#for instance in img_list:
		for index in pbar:
			instance = img_list[index]
			
			if type(instance) == list and len(instance) >= 2:
				img1_path = instance[0]
				img2_path = instance[1]
				
				ensemble_features = []; ensemble_features_string = "["
				
				for i in  model_names:
					custom_model = model[i]
					
					#input_shape = custom_model.layers[0].input_shape[1:3] #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
	
					input_shape = custom_model.layers[0].input_shape
					
					if type(input_shape) == list:
						input_shape = input_shape[0][1:3]
					else:
						input_shape = input_shape[1:3]
					
					
					img1 = functions.preprocess_face(img = img1_path, target_size = input_shape, enforce_detection = enforce_detection, detector_backend = detector_backend)
					img2 = functions.preprocess_face(img = img2_path, target_size = input_shape, enforce_detection = enforce_detection, detector_backend = detector_backend)
					
					img1_representation = custom_model.predict(img1)[0,:]
					img2_representation = custom_model.predict(img2)[0,:]
					
					for j in metrics:
						if j == 'cosine':
							distance = dst.findCosineDistance(img1_representation, img2_representation)
						elif j == 'euclidean':
							distance = dst.findEuclideanDistance(img1_representation, img2_representation)
						elif j == 'euclidean_l2':
							distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
						
						if i == 'OpenFace' and j == 'euclidean': #this returns same with OpenFace - euclidean_l2
							continue
						else:
							
							ensemble_features.append(distance)
							
							if len(ensemble_features) > 1:
								ensemble_features_string += ", "
							ensemble_features_string += str(distance)
							
				#print("ensemble_features: ", ensemble_features)
				ensemble_features_string += "]"
				
				#-------------------------------
				#find newlandface path
				
				home = str(Path.home())
				home = 'D:/newlandface'
				if os.path.isfile(home+'/weights/face-recognition-ensemble-model.txt') != True:
					print("face-recognition-ensemble-model.txt will be downloaded...")
					url = 'https://raw.githubusercontent.com/serengil/newlandface/master/newlandface/models/face-recognition-ensemble-model.txt'
					output = home+'/weights/face-recognition-ensemble-model.txt'
					gdown.download(url, output, quiet=False)
					
				ensemble_model_path = home+'/weights/face-recognition-ensemble-model.txt'
				
				#print(ensemble_model_path)
				
				#-------------------------------
				
				deepface_ensemble = lgb.Booster(model_file = ensemble_model_path)
				
				prediction = deepface_ensemble.predict(np.expand_dims(np.array(ensemble_features), axis=0))[0]
				
				verified = np.argmax(prediction) == 1
				if verified: identified = "true"
				else: identified = "false"
				
				score = prediction[np.argmax(prediction)]
				
				#print("verified: ", verified,", score: ", score)
				
				resp_obj = "{"
				resp_obj += "\"verified\": "+identified
				resp_obj += ", \"score\": "+str(score)
				resp_obj += ", \"distance\": "+ensemble_features_string
				resp_obj += ", \"model\": [\"VGG-Face\", \"Facenet\", \"OpenFace\", \"newlandface\"]"
				resp_obj += ", \"similarity_metric\": [\"cosine\", \"euclidean\", \"euclidean_l2\"]"
				resp_obj += "}"
				
				#print(resp_obj)
				
				resp_obj = json.loads(resp_obj) #string to json
				
				if bulkProcess == True:
					resp_objects.append(resp_obj)
				else:
					return resp_obj
				
				#-------------------------------
		
		if bulkProcess == True:
			resp_obj = "{"

			for i in range(0, len(resp_objects)):
				resp_item = json.dumps(resp_objects[i])

				if i > 0:
					resp_obj += ", "

				resp_obj += "\"pair_"+str(i+1)+"\": "+resp_item
			resp_obj += "}"
			resp_obj = json.loads(resp_obj)
			return resp_obj
		
		return None
		
	#ensemble learning block end
	#--------------------------------
	#ensemble learning disabled
	
	if model == None:
		if model_name == 'VGG-Face':
			print("Using VGG-Face model backend and", distance_metric,"distance.")
			model = VGGFace.loadModel()

		elif model_name == 'OpenFace':
			print("Using OpenFace model backend", distance_metric,"distance.")
			model = OpenFace.loadModel()

		elif model_name == 'Facenet':
			print("Using Facenet model backend", distance_metric,"distance.")
			model = Facenet.loadModel()

		elif model_name == 'newlandface':
			print("Using FB newlandface model backend", distance_metric,"distance.")
			model = FbDeepFace.loadModel()
		
		elif model_name == 'DeepID':
			print("Using DeepID2 model backend", distance_metric,"distance.")
			model = DeepID.loadModel()
		
		elif model_name == 'Dlib':
			print("Using Dlib ResNet model backend", distance_metric,"distance.")
			from newlandface.basemodels.DlibResNet import DlibResNet #this is not a must because it is very huge.
			model = DlibResNet()

		else:
			raise ValueError("Invalid model_name passed - ", model_name)
	else: #model != None
		print("Already built model is passed")

	#------------------------------
	#face recognition models have different size of inputs
	#my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
		
	if model_name == 'Dlib': #this is not a regular keras model
		input_shape = (150, 150, 3)
	
	else: #keras based models
		input_shape = model.layers[0].input_shape
		
		if type(input_shape) == list:
			input_shape = input_shape[0][1:3]
		else:
			input_shape = input_shape[1:3]
	  
	input_shape_x = input_shape[0]
	input_shape_y = input_shape[1]

	#------------------------------

	#tuned thresholds for model and metric pair
	threshold = functions.findThreshold(model_name, distance_metric)

	#------------------------------
	
	#calling newlandface in a for loop causes lots of progress bars. this prevents it.
	disable_option = False if len(img_list) > 1 else True
	
	pbar = tqdm(range(0,len(img_list)), desc='Verification', disable = disable_option)
	
	#for instance in img_list:
	for index in pbar:
	
		instance = img_list[index]
		
		if type(instance) == list and len(instance) >= 2:
			img1_path = instance[0]
			img2_path = instance[1]

			#----------------------
			#crop and align faces

			img1 = functions.preprocess_face(img=img1_path, target_size=(input_shape_y, input_shape_x), enforce_detection = enforce_detection, detector_backend = detector_backend)
			img2 = functions.preprocess_face(img=img2_path, target_size=(input_shape_y, input_shape_x), enforce_detection = enforce_detection, detector_backend = detector_backend)

			#----------------------
			#find embeddings

			img1_representation = model.predict(img1)[0,:]
			img2_representation = model.predict(img2)[0,:]

			#----------------------
			#find distances between embeddings

			if distance_metric == 'cosine':
				distance = dst.findCosineDistance(img1_representation, img2_representation)
			elif distance_metric == 'euclidean':
				distance = dst.findEuclideanDistance(img1_representation, img2_representation)
			elif distance_metric == 'euclidean_l2':
				distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
			else:
				raise ValueError("Invalid distance_metric passed - ", distance_metric)

			#----------------------
			#decision

			if distance <= threshold:
				identified =  "true"
			else:
				identified =  "false"

			#----------------------
			#response object

			resp_obj = "{"
			resp_obj += "\"verified\": "+identified
			resp_obj += ", \"distance\": "+str(distance)
			resp_obj += ", \"max_threshold_to_verify\": "+str(threshold)
			resp_obj += ", \"model\": \""+model_name+"\""
			resp_obj += ", \"similarity_metric\": \""+distance_metric+"\""
			resp_obj += "}"

			resp_obj = json.loads(resp_obj) #string to json

			if bulkProcess == True:
				resp_objects.append(resp_obj)
			else:
				#K.clear_session()
				return resp_obj
			#----------------------

		else:
			raise ValueError("Invalid arguments passed to verify function: ", instance)

	#-------------------------

	toc = time.time()

	#print("identification lasts ",toc-tic," seconds")

	if bulkProcess == True:
		resp_obj = "{"

		for i in range(0, len(resp_objects)):
			resp_item = json.dumps(resp_objects[i])

			if i > 0:
				resp_obj += ", "

			resp_obj += "\"pair_"+str(i+1)+"\": "+resp_item
		resp_obj += "}"
		resp_obj = json.loads(resp_obj)
		return resp_obj
		#return resp_objects













def find(img_path, db_path, model_name ='VGG-Face', distance_metric = 'cosine', model = None, enforce_detection = True, detector_backend = 'opencv'):
	
	model_names = ['VGG-Face', 'Facenet', 'OpenFace', 'newlandface']
	metric_names = ['cosine', 'euclidean', 'euclidean_l2']
	
	tic = time.time()
	
	if type(img_path) == list:
		bulkProcess = True
		img_paths = img_path.copy()
	else:
		bulkProcess = False
		img_paths = [img_path]
	
	if os.path.isdir(db_path) == True:
		
		#---------------------------------------
		
		if model == None:
			if model_name == 'VGG-Face':
				print("Using VGG-Face model backend and", distance_metric,"distance.")
				model = VGGFace.loadModel()
			elif model_name == 'OpenFace':
				print("Using OpenFace model backend", distance_metric,"distance.")
				model = OpenFace.loadModel()
			elif model_name == 'Facenet':
				print("Using Facenet model backend", distance_metric,"distance.")
				model = Facenet.loadModel()
			elif model_name == 'newlandface':
				print("Using FB newlandface model backend", distance_metric,"distance.")
				model = FbDeepFace.loadModel()
			elif model_name == 'DeepID':
				print("Using DeepID model backend", distance_metric,"distance.")
				model = DeepID.loadModel()
			elif model_name == 'Dlib':
				print("Using Dlib ResNet model backend", distance_metric,"distance.")
				from newlandface.basemodels.DlibResNet import DlibResNet #this is not a must because it is very huge
				model = DlibResNet()
			elif model_name == 'Ensemble':
				print("Ensemble learning enabled")
				#TODO: include DeepID in ensemble method
				
				import lightgbm as lgb #lightgbm==2.3.1
				
				models = {}
				
				pbar = tqdm(range(0, len(model_names)), desc='Face recognition models')
				
				for index in pbar:
					if index == 0:
						pbar.set_description("Loading VGG-Face")
						models['VGG-Face'] = VGGFace.loadModel()
					elif index == 1:
						pbar.set_description("Loading FaceNet")
						models['Facenet'] = Facenet.loadModel()
					elif index == 2:
						pbar.set_description("Loading OpenFace")
						models['OpenFace'] = OpenFace.loadModel()
					elif index == 3:
						pbar.set_description("Loading newlandface")
						models['newlandface'] = FbDeepFace.loadModel()
						
			else:
				raise ValueError("Invalid model_name passed - ", model_name)	
		else: #model != None
			print("Already built model is passed")
			
			if model_name == 'Ensemble':
			
				import lightgbm as lgb #lightgbm==2.3.1
				
				#validate model dictionary because it might be passed from input as pre-trained
				
				found_models = []
				for key, value in model.items():
					found_models.append(key)
				
				if ('VGG-Face' in found_models) and ('Facenet' in found_models) and ('OpenFace' in found_models) and ('newlandface' in found_models):
					print("Ensemble learning will be applied for ", found_models," models")
				else:
					raise ValueError("You would like to apply ensemble learning and pass pre-built models but models must contain [VGG-Face, Facenet, OpenFace, newlandface] but you passed "+found_models)
				
				models = model.copy()
		
		#threshold = functions.findThreshold(model_name, distance_metric)
		
		#---------------------------------------
		
		file_name = "representations_%s.pkl" % (model_name)
		file_name = file_name.replace("-", "_").lower()
		
		if path.exists(db_path+"/"+file_name):
			
			print("WARNING: Representations for images in ",db_path," folder were previously stored in ", file_name, ". If you added new instances after this file creation, then please delete this file and call find function again. It will create it again.")
			
			f = open(db_path+'/'+file_name, 'rb')
			representations = pickle.load(f)
			
			print("There are ", len(representations)," representations found in ",file_name)
			
		else:
			employees = []
			
			for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
				for file in f:
					if ('.jpg' in file):
						exact_path = r + "/" + file
						employees.append(exact_path)
			
			if len(employees) == 0:
				raise ValueError("There is no image in ", db_path," folder!")
			
			#------------------------
			#find representations for db images
			
			representations = []
			
			pbar = tqdm(range(0,len(employees)), desc='Finding representations')
			
			#for employee in employees:
			for index in pbar:
				employee = employees[index]
				
				if model_name != 'Ensemble':
				
					if model_name == 'Dlib': #non-keras model
						input_shape = (150, 150, 3)
					else:
						#input_shape = model.layers[0].input_shape[1:3] #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
						
						input_shape = model.layers[0].input_shape
						
						if type(input_shape) == list:
							input_shape = input_shape[0][1:3]
						else:
							input_shape = input_shape[1:3]
					
					#---------------------
					
					input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
					
					img = functions.preprocess_face(img = employee, target_size = (input_shape_y, input_shape_x), enforce_detection = enforce_detection, detector_backend = detector_backend)
					representation = model.predict(img)[0,:]
					
					instance = []
					instance.append(employee)
					instance.append(representation)
					
				else: #ensemble learning
					
					instance = []
					instance.append(employee)
					
					for j in model_names:
						ensemble_model = models[j]
						
						#input_shape = model.layers[0].input_shape[1:3] #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
	
						input_shape = ensemble_model.layers[0].input_shape
						
						if type(input_shape) == list:
							input_shape = input_shape[0][1:3]
						else:
							input_shape = input_shape[1:3]
						
						input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
						
						img = functions.preprocess_face(img = employee, target_size = (input_shape_y, input_shape_x), enforce_detection = enforce_detection, detector_backend = detector_backend)
						representation = ensemble_model.predict(img)[0,:]
						instance.append(representation)
				
				#-------------------------------
				
				representations.append(instance)
			
			f = open(db_path+'/'+file_name, "wb")
			pickle.dump(representations, f)
			f.close()
			
			print("Representations stored in ",db_path,"/",file_name," file. Please delete this file when you add new identities in your database.")
		
		#----------------------------
		#we got representations for database
		
		if model_name != 'Ensemble':
			df = pd.DataFrame(representations, columns = ["identity", "representation"])
		else: #ensemble learning
			df = pd.DataFrame(representations, columns = ["identity", "VGG-Face_representation", "Facenet_representation", "OpenFace_representation", "deepface_representation"])
			
		df_base = df.copy()
		
		resp_obj = []
		
		global_pbar = tqdm(range(0,len(img_paths)), desc='Analyzing')
		for j in global_pbar:
			img_path = img_paths[j]
		
			#find representation for passed image
			
			if model_name == 'Ensemble':
				for j in model_names:
					ensemble_model = models[j]
					
					#input_shape = ensemble_model.layers[0].input_shape[1:3] #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
	
					input_shape = ensemble_model.layers[0].input_shape
					
					if type(input_shape) == list:
						input_shape = input_shape[0][1:3]
					else:
						input_shape = input_shape[1:3]
					
					img = functions.preprocess_face(img = img_path, target_size = input_shape, enforce_detection = enforce_detection, detector_backend = detector_backend)
					target_representation = ensemble_model.predict(img)[0,:]
					
					for k in metric_names:
						distances = []
						for index, instance in df.iterrows():
							source_representation = instance["%s_representation" % (j)]
							
							if k == 'cosine':
								distance = dst.findCosineDistance(source_representation, target_representation)
							elif k == 'euclidean':
								distance = dst.findEuclideanDistance(source_representation, target_representation)
							elif k == 'euclidean_l2':
								distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation), dst.l2_normalize(target_representation))
							
							distances.append(distance)
						
						if j == 'OpenFace' and k == 'euclidean':
							continue
						else:
							df["%s_%s" % (j, k)] = distances
				
				#----------------------------------
				
				feature_names = []
				for j in model_names:
					for k in metric_names:
						if j == 'OpenFace' and k == 'euclidean':
							continue
						else:
							feature = '%s_%s' % (j, k)
							feature_names.append(feature)
				
				#print(df[feature_names].head())
				
				x = df[feature_names].values
				
				#----------------------------------
				#lightgbm model
				home = str(Path.home())
				home = 'D:/newlandface'
				if os.path.isfile(home+'/weights/face-recognition-ensemble-model.txt') != True:
					print("face-recognition-ensemble-model.txt will be downloaded...")
					url = 'https://raw.githubusercontent.com/serengil/newlandface/master/newlandface/models/face-recognition-ensemble-model.txt'
					output = home+'/weights/face-recognition-ensemble-model.txt'
					gdown.download(url, output, quiet=False)
					
				ensemble_model_path = home+'/weights/face-recognition-ensemble-model.txt'
				
				deepface_ensemble = lgb.Booster(model_file = ensemble_model_path)
				
				y = deepface_ensemble.predict(x)
				
				verified_labels = []; scores = []
				for i in y:
					verified = np.argmax(i) == 1
					score = i[np.argmax(i)]
					
					verified_labels.append(verified)
					scores.append(score)
				
				df['verified'] = verified_labels
				df['score'] = scores
				
				df = df[df.verified == True]
				#df = df[df.score > 0.99] #confidence score
				df = df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)
				df = df[['identity', 'verified', 'score']]
				
				resp_obj.append(df)
				df = df_base.copy() #restore df for the next iteration
				
				#----------------------------------
			
			if model_name != 'Ensemble':
				
				if model_name == 'Dlib': #non-keras model
					input_shape = (150, 150, 3)
				else:
					#input_shape = model.layers[0].input_shape[1:3] #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
					
					input_shape = model.layers[0].input_shape
					
					if type(input_shape) == list:
						input_shape = input_shape[0][1:3]
					else:
						input_shape = input_shape[1:3]
				
				#------------------------
				
				input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
				
				img = functions.preprocess_face(img = img_path, target_size = (input_shape_y, input_shape_x), enforce_detection = enforce_detection, detector_backend = detector_backend)
				target_representation = model.predict(img)[0,:]
		
				distances = []
				for index, instance in df.iterrows():
					source_representation = instance["representation"]
					
					if distance_metric == 'cosine':
						distance = dst.findCosineDistance(source_representation, target_representation)
					elif distance_metric == 'euclidean':
						distance = dst.findEuclideanDistance(source_representation, target_representation)
					elif distance_metric == 'euclidean_l2':
						distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation), dst.l2_normalize(target_representation))
					else:
						raise ValueError("Invalid distance_metric passed - ", distance_metric)
					
					distances.append(distance)
				
				threshold = functions.findThreshold(model_name, distance_metric)
				
				df["distance"] = distances
				df = df.drop(columns = ["representation"])
				df = df[df.distance <= threshold]
			
				df = df.sort_values(by = ["distance"], ascending=True).reset_index(drop=True)
				resp_obj.append(df)
				df = df_base.copy() #restore df for the next iteration
			
		toc = time.time()
		
		print("find function lasts ",toc-tic," seconds")
		
		if len(resp_obj) == 1:
			return resp_obj[0]
		
		return resp_obj
		
	else:
		raise ValueError("Passed db_path does not exist!")
		
	return None
	
def stream(db_path = '', model_name ='VGG-Face', distance_metric = 'cosine', enable_face_analysis = True):
	realtime.analysis(db_path, model_name, distance_metric, enable_face_analysis)

def allocateMemory():
	print("Analyzing your system...")
	functions.allocateMemory()

#---------------------------
#main

functions.initializeFolder()

