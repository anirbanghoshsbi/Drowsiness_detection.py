# usage detect.py -ln shape_predictor_68_face_landmark.dat --video video.mov
#As learnt from Dr. Adrian Rosebrock...
# import packages
from scipy.spatial import distance
from imutils import face_utils  # handy function that comes with imutils can be used for converting image to array
import imutils
import dlib
import cv2
import argparse

# define a function to to detect the aspect ratio between the two eyes
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
# we would use the dlib library frontal face detector and facial shape predictor having 68 key facial landmark stored in dat file	
ap =argparse.ArgumentParser()
ap.add_argument('-ln','--landmark',required ='True', help = 'path to the shape_predictor_68_face_landmarks.dat file')
ap.add_argument('-v','--video',help = 'path to the video')
args =vars(ap.parse_args())
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(args['landmark']) 
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# if video reference is not provided then capture it from the webcam
if not args.get("video", False):
    camera=cv2.VideoCapture(0)
else :
    camera = cv2.VideoCapture(args["video"])
flag=0
while True:
	ret, frame=camera.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)         #converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
camera.stop()
