import cv2
import numpy as np
import os

os.chdir("/home/sathya-prakash/Desktop/gitrepo/Haar_Cascade")

hand_cascade = cv2.CascadeClassifier('haarcascade_palm.xml')# detects both hands

cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in hands:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()