import cv2
import sys
import os
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from matplotlib import pyplot as plt
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import face_recognition
import img
from pygame import mixer
from gtts import gTTS 

facePath = '/Users/rraina/Desktop/INTERNSHIP 2019/FacialReg/haarcascade_frontalface_alt.xml'
eyePath = '/Users/rraina/Desktop/INTERNSHIP 2019/FacialReg/haarcascade_lefteye_2splits.xml'
faceCascade = cv2.CascadeClassifier(facePath)
eyeCascade = cv2.CascadeClassifier(eyePath)


video_capture = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

Ritik_came = False

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    small_frame = cv2.resize(frame, (0, 0), fx=1.00, fy=1.00)

    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        img = cv2.rectangle(small_frame, (x, y), (x+w, y+h), (255, 0, 0), 1)


    # Display the resulting frame
    cv2.imshow('Video', small_frame)
    
    for n in range(0, len(faces)):
        image1 = img[y:faces[n][1] + faces[n][3], x:faces[n][0] + faces[n][2]]
        image2 = face_recognition.load_image_file("/Users/rraina/Desktop/INTERNSHIP 2019/FacialReg/Training Images/image.jpg") 
        if (len(image1) > 0):
            if(len(face_recognition.face_encodings(image1)) > 0):
                encoding_1 = face_recognition.face_encodings(image1)[0]
                encoding_2 = face_recognition.face_encodings(image2)[0]
                results_Ritik = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.70)
                if ((results_Ritik[0] == True) and (Ritik_came == False)):
                    Ritik_came = True
                    mytext = 'Welcome Mr. Ritik Raina'
                    language = 'en'
                    myobj = gTTS(text=mytext, lang=language, slow=False)
                    myobj.save("welcome.mp3")
                    mixer.init()
                    mixer.music.load('/Users/rraina/Desktop/INTERNSHIP 2019/FacialReg/welcome.mp3')
                    mixer.music.play()
                    said = False
    said = True            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
