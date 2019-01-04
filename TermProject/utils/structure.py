#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : structure
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-12-30
'''

import os
import numpy as np
import cv2

def fundamentalMat_estimation(findFundamentalMat, pts1, pts2):
    """Perform fundamental matrix estimation.
    Parameters
    ----------
    findFundamentalMat : lambada with a signature lambada pts1,pts2,method : F,mask
        accepts two types of lambada:
            opencv implementation : 
                lambda pts1, pts2, method : cv2.findFundamentalMat(pts1, pts2, method, 1.0, 0.99)
            custom implementation : 
                lambda pts1, pts2, method : m_methods.m_findFundamentalMat(pts1, pts2, method, 10000.0, 0.99)
    pts1 : ndarray Nx2, dtype = float
        contains points in the reference view. 
    pts2 : ndarray Nx2, dtype = float
        contains points in the other view.
    Return
    ------
    pts1 : ndarray Mx2, dtype = float
        inlier points in the reference view. 
    pts2 : ndarray Mx2, dtype = float
        inlier points points in the other view.
    F : ndarray 3x3, dtype = float
        output fundamental matrix.
    """

    # Find the Fundamental Matrix via RANSAC
    F, mask = findFundamentalMat(pts1, pts2, cv2.FM_8POINT + cv2.RANSAC)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = pts1[mask==1]
    pts2 = pts2[mask==1]

    # Find the Fundamental Matrix from all inlier points
    F, mask = findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = pts1[mask==1]
    pts2 = pts2[mask==1]

    return pts1, pts2, F


def compute_extrinsic_from_essential(E):
    """Computes extrinsic matrix of the camera from essential matrix.
    Parameters
    ----------
    F : ndarray 3x3, dtype = float
        the essential matrix.
    Return
    ------
    P2s : list containing 4 ndarray of shape (3, 4)
        4 possible projection matrix solutions.
    """

    # Perform SVD
    U, __, VT = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, VT)) < 0:
        VT = -VT

    # create 4 possible projection matrice (Hartley p 258)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    skew_t = np.dot(np.dot(U, [[0,1,0],[-1,0,0],[0,0,0]]), U.T)
    t = np.array([[-skew_t[1,2],skew_t[0,2],-skew_t[0,1]]]).T
    P2s = [np.hstack((np.dot(U, np.dot(W, VT)), t)),
            np.hstack((np.dot(U, np.dot(W, VT)), -t)),
            np.hstack((np.dot(U, np.dot(W.T, VT)), t)),
            np.hstack((np.dot(U, np.dot(W.T, VT)), -t))]
    return P2s

def find_correct_projection(P2s, intrinsic_matrix, P1, pt1, pt2):
    """Computes extrinsic matrix of the camera from essential matrix.
    Parameters
    ----------
    P2s : list containing 4 ndarray of shape (3, 4)
        4 possible projection matrix solutions of the other view.
    intrinsic_matrix : ndarray 3x3, dtype = float
        the intrinsic matrix of the camera.
    P1 : ndarray 3x4, dtype = float
        the projection matrix of the reference view.
    pt1 : ndarray 1x2, dtype = float
        one sample point in the reference view. 
    pt2 : ndarray 1x2, dtype = float
        one sample point in the other view.
    Return
    ------
    P2 : ndarray 3x4, dtype = float
        the correct projection matrix of the other view.
    """

    # Find the correct extrinsic matrix
    ind = -1
    for i, P2 in enumerate(P2s):
        P2 = np.dot(intrinsic_matrix, P2)
        
        # triangulate the sample point
        d1 = cv2.triangulatePoints(P1, P2, pt1, pt2)
        d1 /= d1[3]

        # convert P2 from camera view to world view
        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)

        if d1[2] > 0 and d2[2] > 0:
            ind = i

    return np.dot(intrinsic_matrix, P2s[ind])


def find_common_keypoints(pts1, pts2):

    mask = -np.ones((len(pts2))).astype('int')
    for i, row in enumerate(pts2, 1):
        idx = np.where((pts1 == row).all(1))[0]
        if idx.size:
            mask[i - 1] = idx[0]
    return mask


def find_3Dto2D_point_correspondences(pts1, pts2, pts3D, pts_ref):

    # Search common keypoints
    mask = find_common_keypoints(pts_ref, pts1)

    # Determine 3D-2D point correspondences on the third view
    map_2Dto3D = {idx2D - 1:idx3D for idx2D, idx3D in enumerate(mask, 1) if idx3D != -1}
    return pts2[list(map_2Dto3D.keys())], pts3D[list(map_2Dto3D.values())]
