#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles

Licensed under the Apache License, Version 2.0
'''

import os
import numpy as np
import cv2
import glob

def camera_calibration(input_pattern_path, resolution = [1280, 960], number_chessboard_colum = 6, number_chessboard_row = 8, output_images_path = None, output_camera_para_filename = 'camera_parameters.npz'):

    if output_images_path is not None:
        if not os.path.exists(output_images_path):
            os.makedirs(output_images_path)

    # Define resolution of input images
    RESOLUTION_X = resolution[0]
    RESOLUTION_Y = resolution[1]

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((number_chessboard_colum*number_chessboard_row, 3), np.float32)
    objp[:,:2] = np.mgrid[0:number_chessboard_row, 0:number_chessboard_colum].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Define a file manager
    images = []
    for ext in ('*.jpg', '*.JPG', '*.png', '*.PNG', '*.tif', '*.TIF', '*.bmp', '*.BMP'):
        images.extend(glob.glob(os.path.join(input_pattern_path, ext)))
    for fname in images:
        print('Processing {}..'.format(fname.split(os.sep)[-1]))
        
        img = cv2.imread(fname)
        img = cv2.resize(img, (RESOLUTION_X, RESOLUTION_Y))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (number_chessboard_row,number_chessboard_colum), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and save the corners
            if output_images_path is not None:
                img = cv2.drawChessboardCorners(img, (number_chessboard_row,number_chessboard_colum), corners2, ret)
                cv2.imwrite(os.path.join(output_images_path, fname.split(os.sep)[-1].replace('.', '_corners.')), img)
        else:
            print('Fail to find corners.')

    # Calibration
    ret, camera_matrix, distortion_vector, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Calculate reprojection error
    reprojection_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, distortion_vector)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        reprojection_error += error
    reprojection_error /= len(objpoints)

    if output_images_path is not None:
        np.savez(os.path.join(output_images_path, output_camera_para_filename), 
            intrinsic_matrix = camera_matrix, distortion_vector = distortion_vector, reprojection_error = reprojection_error) 

    return camera_matrix, distortion_vector, reprojection_error

def undistort_images(images_path, images_base, camera_matrix, distortion_vector, resolution = [1280, 960], output_path = None):

    # Define resolution of input images
    RESOLUTION_X = resolution[0]
    RESOLUTION_Y = resolution[1]

    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    # Define a file manager
    images = []
    for ext in ('*.jpg', '*.JPG', '*.png', '*.PNG', '*.tif', '*.TIF', '*.bmp', '*.BMP'):
        images.extend(glob.glob(os.path.join(images_path, ext)))
    for fname in images:
        if not fname.split(os.sep)[-1].startswith(images_base):
            continue
        print('Processing {}..'.format(fname.split(os.sep)[-1]))

        img = cv2.imread(fname)
        img = cv2.resize(img, (RESOLUTION_X, RESOLUTION_Y))

        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_vector, (w,h), 1, (w,h))

        # Undistort using mapping
        mapx,mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_vector, None, new_camera_matrix, (w,h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        if output_path is not None:
            cv2.imwrite(os.path.join(output_path, fname.split(os.sep)[-1].split('.')[0] + '.jpg'), dst)
