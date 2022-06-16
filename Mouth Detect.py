#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:18:02 2022

@author: dhruviljhala
"""

import cv2
import numpy as np

mouth_cascade = cv2.CascadeClassifier('./haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

cap = cv2.VideoCapture("./videoplayback(40s).mp4")
ds_factor = 1

imageWidth = int(cap.get(3))
imageHeight = int(cap.get(4))
fps = 30#cap.get(cv2.CAP_PROP_FPS)
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')

writer= cv2.VideoWriter('outpy.AVI',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (imageWidth, imageHeight), 0)


while True:
    ret, frame = cap.read()
    if ret == True:

        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
        for (x,y,w,h) in mouth_rects:
            y = int(y - 0.15*h)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
            break
        
        writer.write(frame)
        cv2.imshow('Mouth Detector', frame)
    
    
        c = cv2.waitKey(1)
        if c == 27:
            break
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()






    
    
