#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:22:36 2019

Extract person image samples from video folder into Sample folder.
For use to make recognition sample.

@author: gjz
"""
import os
import cv2
import numpy as np

from yolo import YOLO

VID_FOLDER = './Video'
SAVE_FOLDER = './Sample/ALL'

crop_every_frame = 5

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

yolo = YOLO()

vid_list = os.listdir(VID_FOLDER)
for vid in vid_list:
    print('Process on : ' + vid)
    cap = cv2.VideoCapture(os.path.join(VID_FOLDER, vid))
    vid_name = os.path.splitext(vid)[0]
    c = 1
    f = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if f % crop_every_frame == 0:
                boxs, clss, scrs = yolo.predict(frame, filters=['person'])
                
                for box in boxs:
                    x, y, w, h = box
                    
                    top = max(0, np.floor(x + 0.5).astype(int))
                    left = max(0, np.floor(y + 0.5).astype(int))
                    right = min(frame.shape[1], np.floor(x + w + 0.5).astype(int))
                    bottom = min(frame.shape[0], np.floor(y + h + 0.5).astype(int))
                    
                    img = frame[left:bottom, top:right]
                    
                    file_name = vid_name + '_' + str(c) + '.jpg'
                    cv2.imwrite(os.path.join(SAVE_FOLDER, file_name), img)
                    c += 1
            f += 1
        else:
            break
    
    cap.release()
