#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : bundle_adjustment
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-12-30
'''

import os
import numpy as np
import cv2
import glob
from scipy.optimize import least_squares

def reprojection_error(objpoints, imgpoints, projection_matrix, distortion_vector):

    # Decompose projection matrix to get camera matrix
    camera_matrix = cv2.decomposeProjectionMatrix(projection_matrix)[0]
    # Calculate extrinsic matrix ([R|t]) from camera matrix and projection matrix
    extrinsic_matrix = np.dot(np.linalg.inv(camera_matrix), projection_matrix)
    R, t = extrinsic_matrix[:3, :3], extrinsic_matrix[:, 3]

    # Calculate reprojection error
    imgpoints2 = cv2.projectPoints(objpoints, cv2.Rodrigues(R)[0], t, camera_matrix, distortion_vector)[0]
    # the return array of cv2.projectPoints shapes as (N, 1, 2)
    imgpoints1 = np.asarray([[row.tolist()] for row in imgpoints])
    error = cv2.norm(imgpoints1, imgpoints2, cv2.NORM_L2) / len(imgpoints2)

    return error

def bundle_adjustment():

    # TODO

    return
