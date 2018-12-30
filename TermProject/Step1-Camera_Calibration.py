#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : Camera Calibration
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-12-24
'''

import os
import numpy as np
import cv2
import glob

input_path = 'chessboard_pattern'
output_path = os.path.join('.tmp','calibration_result')

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Define specification of the chessboard
NUMBER_COLUMN = 6
NUMBER_ROW = 8

# Define resolution of input images
RESOLUTION_X = 1280
RESOLUTION_Y = 960

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((NUMBER_COLUMN*NUMBER_ROW,3), np.float32)
objp[:,:2] = np.mgrid[0:NUMBER_ROW,0:NUMBER_COLUMN].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Define a file manager
images = []
for ext in ('*.jpg', '*.JPG', '*.png', '*.PNG'):
    images.extend(glob.glob(os.path.join(input_path, ext)))
for fname in images:
    print('Processing {}..'.format(fname.split(os.sep)[-1]))
    
    img = cv2.imread(fname)
    img = cv2.resize(img, (RESOLUTION_X, RESOLUTION_Y))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (NUMBER_ROW,NUMBER_COLUMN), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and save the corners
        img = cv2.drawChessboardCorners(img, (NUMBER_ROW,NUMBER_COLUMN), corners2, ret)
        cv2.imwrite(os.path.join(output_path, fname.split(os.sep)[-1].replace('.', '_corners.')), img)
    else:
        print('Fail to find corners.')

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Calculate reprojection error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error
print('total error: {}'.format(total_error/len(objpoints)))
print('Camera matrix: \n{}'.format(mtx))
print('Dist matrix: \n{}'.format(dist))

np.savez(os.path.join(output_path, 'camera_parameters.npz'), mtx=mtx, dist=dist)
