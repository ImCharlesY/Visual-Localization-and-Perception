#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : feature
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-12-30
'''

import os
import numpy as np
import cv2

def match_keypoints_between_images(img1, img2, output_path = None, inlier_points_filename = 'inliers.npz'):
   
    # find the keypoints and descriptors with SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)

    pts1 = np.asarray([kp1[m.queryIdx].pt for m in good]).astype('float64')
    pts2 = np.asarray([kp2[m.trainIdx].pt for m in good]).astype('float64')

    # Constrain matches to fit homography
    __, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = pts1[mask == 1]
    pts2 = pts2[mask == 1]  

    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.savez(os.path.join(output_path, inlier_points_filename), pts1=pts1, pts2=pts2)

    return pts1, pts2
