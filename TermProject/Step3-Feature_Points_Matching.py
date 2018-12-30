#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : Feature Points Matching
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-12-24
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

# If run in Windows, comment this line.
plt.switch_backend('agg')

input_path = os.path.join('.tmp','undistorted_images')
output_path = os.path.join('.tmp','feature_points_matching_result')
image_file_1 = 'adminB2_01.jpg'
image_file_2 = 'adminB2_02.jpg'
image_file_3 = 'adminA_03.jpg'

# Define resolution of input images
RESOLUTION_X = 1280
RESOLUTION_Y = 960

if not os.path.exists(output_path):
    os.makedirs(output_path)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


img1 = cv2.imread(os.path.join(input_path, image_file_1),0)  #queryimage # left image
img1 = cv2.resize(img1, (RESOLUTION_X, RESOLUTION_Y))
img2 = cv2.imread(os.path.join(input_path, image_file_2),0) #trainimage # right image
img2 = cv2.resize(img2, (RESOLUTION_X, RESOLUTION_Y))

# find the keypoints and descriptors with SIFT
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        good.append(m)

pts1 = np.asarray([kp1[m.queryIdx].pt for m in good]).astype('float64')
pts2 = np.asarray([kp2[m.trainIdx].pt for m in good]).astype('float64')

# Constrain matches to fit homography
retval, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 100.0)
mask = mask.ravel()

# We select only inlier points
pts1 = pts1[mask == 1]
pts2 = pts2[mask == 1]

# Find the Fundamental Matrix
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC,3.0,0.99)
mask = mask.ravel()

# We select only inlier points
pts1 = pts1[mask==1]
pts2 = pts2[mask==1]

# Find the Fundamental Matrix
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
mask = mask.ravel()

# We select only inlier points
pts1 = pts1[mask==1]
pts2 = pts2[mask==1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

cv2.imwrite(os.path.join(output_path, image_file_1), img5)
cv2.imwrite(os.path.join(output_path, image_file_2), img3)

print('Fundamental matrix: \n{}'.format(F))

np.savez(os.path.join(output_path, 'fundamental_matrix.npz'), F=F)
np.savez(os.path.join(output_path, 'inlier_pts.npz'), pts1=pts1, pts2=pts2)
