import cv2
import numpy as np
import os

os.chdir("/home/sathya-prakash/Desktop/gitrepo/Haar_Cascade")

palm_cascade = cv2.CascadeClassifier('haarcascade_palm.xml')# detects both hands
#palm_cascade = cv2.CascadeClassifier('palm.xml')
fist_cascade = cv2.CascadeClassifier('fist.xml')#detects fist of right hand only

cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    palms = palm_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in palms:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    fists = fist_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in fists:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()