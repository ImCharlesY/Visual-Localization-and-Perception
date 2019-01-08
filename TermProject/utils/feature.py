#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles

Licensed under the Apache License, Version 2.0
'''

import os
import numpy as np
import cv2

def match_keypoints_between_images(img1, img2, output_path = None, inlier_points_filename = 'inliers.npz'):
   
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    good = []
    pts1 = []
    pts2 = []

    # find the keypoints and descriptors with SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = flann.knnMatch(des1, des2, k=2)

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    # find the keypoints and descriptors with SURF
    surf = cv2.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
    matches = flann.knnMatch(des1, des2, k=2)

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.array(pts1).astype('float64')
    pts2 = np.array(pts2).astype('float64')

    # # Constrain matches to fit homography
    # __, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 10.0)
    # mask = mask.ravel()

    # # We select only inlier points
    # pts1 = pts1[mask == 1]
    # pts2 = pts2[mask == 1]  

    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.savez(os.path.join(output_path, inlier_points_filename), pts1=pts1, pts2=pts2)

    return pts1, pts2
