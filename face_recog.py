import cv2
import sys
import os
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import face_recognition

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

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

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
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Ritik", (x, y-10), font, 0.7, (100,255,100), 1)


    # Display the resulting frame
    cv2.imshow('Video', small_frame)
    path = '/Users/rraina/Desktop/INTERNSHIP 2019/FacialReg/faces'
    os.chdir(path)
    total = len(faces)
    print("Faces Detected: " + str(total))
    for n in range(0, total):
        img = cv2.rectangle(small_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = img[y:faces[n][1] + faces[n][3], x:faces[n][0] + faces[n][2]]
        name = "face" + str(n) + ".jpg"
        cv2.imwrite(name, face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
