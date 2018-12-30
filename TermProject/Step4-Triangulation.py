#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : Triangulation
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-12-29
'''

import os
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# If run in Windows, comment this line.
# plt.switch_backend('agg')

camera_parameter_path = os.path.join('.tmp','calibration_result','camera_parameters.npz')
fundamental_matrix_path = os.path.join('.tmp','feature_points_matching_result','fundamental_matrix.npz')
inlier_pts_path = os.path.join('.tmp','feature_points_matching_result','inlier_pts.npz')
input_path = 'raw_images'
output_path = os.path.join('.tmp','triangulation')

# Define resolution of input images
RESOLUTION_X = 1280
RESOLUTION_Y = 960

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load camera parameter
with np.load(camera_parameter_path) as reader:
    mtx, dist = reader['mtx'], reader['dist']

# Load fundamental matrix
with np.load(fundamental_matrix_path) as reader:
    F = reader['F']

# Load inlier points
with np.load(inlier_pts_path) as reader:
    pts1, pts2 = reader['pts1'].T, reader['pts2'].T

# Calculate essential matrix
E = np.dot(np.dot(mtx.T, F), mtx)

# Decompose essential matrix into rotation matrix and translation vector
def compute_P_from_essential(E):
    """ Compute the second camera matrix (assuming P1 = [I 0])
        from an essential matrix. E = [t]R
    :returns: list of 4 possible camera matrices.
    """
    U, S, VT = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, VT)) < 0:
        VT = -VT

    # create 4 possible camera matrices (Hartley p 258)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    skew_t = np.dot(np.dot(U, [[0,1,0],[-1,0,0],[0,0,0]]), U.T)
    t = np.array([[-skew_t[1,2],skew_t[0,2],-skew_t[0,1]]]).T
    P2s = [np.hstack((np.dot(U, np.dot(W, VT)), t)),
            np.hstack((np.dot(U, np.dot(W, VT)), -t)),
            np.hstack((np.dot(U, np.dot(W.T, VT)), t)),
            np.hstack((np.dot(U, np.dot(W.T, VT)), -t))]
    return P2s


# Calculate projection matrix of the first view
P1 = np.dot(mtx, np.hstack((np.eye(3), np.zeros((3,1)))))

print('Projection matrix of the first view: \n{}'.format(P1))

# Calculate projection matrix of the second view
P2s = compute_P_from_essential(E)
ind = -1
for i, P2 in enumerate(P2s):
    P2 = np.dot(mtx, P2)
    # Find the correct camera parameters
    d1 = cv2.triangulatePoints(P1, P2, pts1[:, 0], pts2[:, 0])
    d1 /= d1[3]

    # Convert P2 from camera view to world view
    P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
    d2 = np.dot(P2_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i
P2 = np.dot(mtx, P2s[ind])

print('Projection matrix of the second view: {} \n{}'.format(ind, P2))

# Triangulation
# ATTENTION: all input data should be of float type or it will raise BUS ERROR exception
pts4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
pts4D /= pts4D[3]

fig = plt.figure()
fig.suptitle('3D reconstructed', fontsize=16)
ax = fig.gca(projection='3d')
ax.scatter(pts4D[0,:], pts4D[1,:], pts4D[2,:], c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# plt.savefig(os.path.join(output_path, '3D_pts_cloud.svg'), bbox_inches = 'tight', format = 'svg') 
plt.show()
