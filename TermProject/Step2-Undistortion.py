#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : Undistortion
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-12-24
'''

import os
import numpy as np
import cv2
import glob

camera_parameter_path = os.path.join('.tmp','calibration_result','camera_parameters.npz')
input_path = 'raw_images'
output_path = os.path.join('.tmp','undistorted_images')

# Define resolution of input images
RESOLUTION_X = 1280
RESOLUTION_Y = 960

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load camera parameter
with np.load(camera_parameter_path) as reader:
    mtx, dist = reader['mtx'], reader['dist']

# Define a file manager
images = []
for ext in ('*.jpg', '*.JPG', '*.png', '*.PNG'):
    images.extend(glob.glob(os.path.join(input_path, ext)))
for fname in images:
    print('Processing {}..'.format(fname.split(os.sep)[-1]))

    img = cv2.imread(fname)
    img = cv2.resize(img, (RESOLUTION_X, RESOLUTION_Y))

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort using mapping
    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(output_path, fname.split(os.sep)[-1]), dst)
