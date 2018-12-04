#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : icp
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date     		: 2018-12-04
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors
from absolute_orientation import *

def best_fit_transform(src, dst):
    """Calculates the least-squares best-fit transform between corresponding 3D points A->B.
    This function uses absolute orientation algorithm.
    Parameters
    ----------
    src : ndarray Nx3
        source 3D points cloud
    dst : ndarray Nx3
        destination 3D points cloud
    Return
    ------
    T : ndarray 4x4
        homogeneous transformation matrix
    R : ndarray 3x3
        rotation matrix
    t : ndarray 3x1
        column vector for translation
    """

    # translate points to their centers
    center_src = np.mean(src, axis=0)
    center_dst = np.mean(dst, axis=0)
    src_centralized = src - center_src
    dst_centralized = dst - center_dst

    # calculate rotation matrix via absolute orientation algorithm
    __, R, __ = absolute_orientation(src_centralized.T, dst_centralized.T)

    # calculate translation vector
    t = center_dst.T - np.dot(R, center_src.T)

    # homogeneous transformation
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, R, t

def nearest_neighbor(src, dst):
    """Find the nearest (Euclidean) neighbor in dst for each point in src via KNN algorithm.
    Parameters
    ----------
    src : ndarray Nx3 
        source 3D points cloud
    dst : ndarray Nx3 
        destination 3D points cloud
    Return
    ------
    dist : ndarray
        Euclidean distances of the nearest neighbor
    indices : ndarray
        dst indices of the nearest neighbor
    """

    dist, indices = NearestNeighbors(n_neighbors=1).fit(dst).kneighbors(src, return_distance=True)
    return dist.ravel(), indices.ravel()

def icp(A, B, maxits=100, tolerance=1e-10):
    """The Iterative Closest Point method.
    Parameters
    ----------
    A : ndarray Nx3
        source 3D points cloud
    B : ndarray Nx3
        destination 3D points cloud
    maxits : int
        maximum number of iterations for the algorithm to perform.
    tolerance : float
        convergence criteria
    Return
    ------
    T : ndarray 4x4
        final homogeneous transformation
    dist : ndarray
        Euclidean distances (errors) of the nearest neighbor
    """

    src = np.ones((4,A.shape[0]))
    dst = np.ones((4,B.shape[0]))
    src[:3,:] = np.copy(A.T)
    dst[:3,:] = np.copy(B.T)

    prev_error = 0
    for i in range(maxits):
        # Step 1 - Matching: find the closest point as the corresponding point using the current alignment
        dist, indices = nearest_neighbor(src[:3,:].T, dst[:3,:].T)

        # Step 2 - Updating: compute the alignment using the close-form solution as introduced previously
        T, _, _ = best_fit_transform(src[:3,:].T, dst[:3,indices].T)
        src = np.dot(T, src)

        # check if converge
        error = np.sum(dist) / dist.size
        if abs(prev_error-error) < tolerance:
            break
        prev_error = error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:3,:].T)

    return T, dist
    