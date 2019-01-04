#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles

Licensed under the Apache License, Version 2.0
'''

import cv2
import numpy as np

def m_ransac(fit_model, validate_model, X, num_samples, max_iter = -1, thresh = 1.0, ratio_of_inliers = 0.99):
    """General implementation of the RANSAC method
    Parameters
    ----------
    fit_model : callable
        function that fits the model, with th signature fun(x)->model.
        The argument x passed to this function is an ndarray of shape (num_samples, m),
        where num_samples is the number of sampling points per iteration in the RANSAC method,
        and m is the dimension of the features of each point. This function should explain the
        meaning of these features itself. It must return the model parameters in only one 
        object (no matter what type).     
    validate_model : callable
        function that computes the errors of the model on all points, with th signature fun(model, x)->error.
        The argument x passed to this function is an ndarray of shape (n, m), where n is 
        the number of the whole samples set, and m is the dimension of the features of 
        each point. This function should explain the meaning of these features itself. 
        It must return the errors as an ndarray of shape (n, ). 
    X : ndarray NxM, dtype = float
        contains a set of observed data points, where N is the number of the whole samples set, 
        and M is the dimension of the features of each point. This function does not care about 
        the meaning of M. You should explain it yourself in function 'fit_model' and function 
        'validate_model'. N must larger than or equal to 8. 
    num_samples : int
        the number of sampling points per iteration.
    max_iter : int
        maximum number of iterations to perform.
    thresh : float
        threshold to distinguish inliers and outliers.
    ratio_of_inliers : float
        when the ratio of inliers exceeds this value, the iteration will stop.
    Return
    ------
    retval : boolen
        whether we find the best model parameters
    best_model : 
        model parameters which best fit the data (or None if no good model is found)
    best_mask : ndarray (N,), dtype = int
        output array of N elements, every element of which is set to 0 for outliers and to 1 for the other points. 
    """

    if len(X) < 8:
        raise ValueError("Number of points must larger than or equal to 8.")

    best_model = None
    best_mask = []
    best_ratio = -1.0

    if max_iter == -1:
        # we use 1.6 times of theoretical maximum number of iterations
        max_iter = int(np.log10(1 - ratio_of_inliers) / np.log10(1 - np.power(0.8, 8)) * 1.6)

    # perform RANSAC iterations
    for it in range(max_iter):

        # sampling randomly
        all_indices = np.arange(X.shape[0])
        np.random.shuffle(all_indices)
     
        indices_1 = all_indices[:num_samples]
        indices_2 = all_indices[num_samples:]
     
        sample_points = X[indices_1,:]
     
        # fit a model for sample points
        model = fit_model(sample_points)

        if model is None:
            continue
     
        # compute error of the model on the whole point cloud   
        dist = validate_model(model, X)
        mask = np.zeros(len(X))
        mask[np.abs(dist) < thresh] = 1

        # cache the best model
        if np.count_nonzero(mask) / len(X) > best_ratio:
            best_ratio = np.count_nonzero(mask) / len(X)
            best_model = model
            best_mask = mask
     
        # done in case we have enough inliers
        if np.count_nonzero(mask) > len(X) * ratio_of_inliers:
            break

    return best_model is not None, best_model, best_mask



""" ---------------------------------- Fundamental Matrix Estimation ----------------------------------------- """

def eight_point_algorithm(pts1, pts2):
    """Performs 8-point algorithm.
    Parameters
    ----------
    pts1 : ndarray Nx3, dtype = float
        contains points in the reference view in homogeneous space.
    pts2 : ndarray Nx3, dtype = float
        contains points in the other view in homogeneous space.
    Return
    ------
    F : ndarray 3x3, dtype = float
        output fundamental matrix.
    """
    
    if pts1.shape[0] != pts1.shape[0]:
        raise ValueError("Number of points don't match.")

    # [x'*x, x'*y, x'*z, y'*x, y'*y, y'*z, z'*x, z'*y, z'*z]
    A = np.vstack([
        pts1[:,0]*pts2[:,0], pts1[:,0]*pts2[:,1], pts1[:,0]*pts2[:,2], 
        pts1[:,1]*pts2[:,0], pts1[:,1]*pts2[:,1], pts1[:,1]*pts2[:,2], 
        pts1[:,2]*pts2[:,0], pts1[:,2]*pts2[:,1], pts1[:,2]*pts2[:,2] ]).T

    # compute linear least square solution
    __, S, VT = np.linalg.svd(A)
    # solution can be obtained from the vector corresponds to the minimum singular value
    F = VT[-1].reshape(3,3)
        
    # constrain F : making rank 2 by zeroing out last singular value
    U, S, VT = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(np.dot(U, np.diag(S)), VT)
    
    return F / F[2,2]


def find_fundamental_matrix(points):
    """Computes the fundamental matrix using 8-point algorithm.
    Parameters
    ----------
    points : ndarray Nx4, dtype = float
        contains a set of observed data points. 
        points[:,0:2] contains points in the reference view.
        points[:,2:4] contains points in the other view.
    Return
    ------
    F : ndarray 3x3, dtype = float
        output fundamental matrix.
    """

    pts1 = np.hstack([points[:,:2], np.ones(len(points)).reshape(-1,1)])
    pts2 = np.hstack([points[:,2:], np.ones(len(points)).reshape(-1,1)])

    # calculate the normalizing transformations for each of the point sets:
    # after the transformation each set will have the mass center at the coordinate origin
    # and the average distance from the origin will be ~sqrt(2).
    mean1 = np.mean(pts1[:,:2], axis = 0)
    S1 = np.sqrt(2) / np.std(pts1[:,:2])
    T1 = np.array([[S1,0,-S1*mean1[0]], [0,S1,-S1*mean1[1]], [0,0,1]])
    pts1 = np.dot(T1,pts1.T).T
    
    mean2 = np.mean(pts2[:,:2], axis = 0)
    S2 = np.sqrt(2) / np.std(pts2[:,:2])
    T2 = np.array([[S2,0,-S2*mean2[0]], [0,S2,-S2*mean2[1]], [0,0,1]])
    pts2 = np.dot(T2,pts2.T).T

    F = eight_point_algorithm(pts1, pts2)

    # reverse normalization
    return np.dot(np.dot(T1.T, F), T2)


def compute_fundamental_matrix_error(mat, points):
    """Computes the fundamental matrix using 8-point algorithm.
    Reference : 
        https://github.com/opencv/opencv/blob/8f15a609afc3c08ea0a5561ca26f1cf182414ca2/modules/calib3d/src/fundam.cpp
    Parameters
    ----------
    mat : ndarray 3x3, dtype = float
        the fundamental matrix.
    points : ndarray Nx4, dtype = float
        contains a set of observed data points. 
        points[:,0:2] contains points in the reference view.
        points[:,2:4] contains points in the other view.
    Return
    ------
    error : ndarray (N,), dtype = float
        contains the distance from points to their epipolar lines.
    """

    pts1 = np.hstack([points[:,:2], np.ones(len(points)).reshape(-1,1)])
    pts2 = np.hstack([points[:,2:], np.ones(len(points)).reshape(-1,1)])

    m2 = np.dot(mat, pts1.T)
    s2 = 1 / (m2[0]**2 + m2[1]**2)
    d2 = (pts2.T * m2).sum(axis = 0)
    err2 = d2**2 * s2

    m1 = np.dot(mat, pts2.T)
    s1 = 1 / (m1[0]**2 + m1[1]**2)
    d1 = (pts1.T * m1).sum(axis = 0)
    err1 = d1**2 * s1

    return np.where(err1 > err2, err1, err2)


def m_findFundamentalMat(pts1, pts2, method, thresh = 10000.0, ratio_of_inliers = 0.99):
    """Commpute the fundamental matrix using the 8-point algorithm or RANSAC + 8-point algorithm.
    Parameters
    ----------
    pts1 : ndarray Nx2, dtype = float
        contains points in the reference view. 
    pts2 : ndarray Nx2, dtype = float
        contains points in the other view.
    method : enumerate {cv2.FM_8POINT, cv2.RANSAC}
        cv2.FM_8POINT for an 8-point algorithm, N should be larger than or equal to 8.
        cv2.RANSAC for the RANSAC algorithm, N should be larger than or equal to 8.
    thresh : float
        parameter used for RANSAC. It is the maximum distance from a point to an epipolar line in pixels, 
        beyond which the point is considered an outlier and is not used for computing the final fundamental 
        matrix.
    ratio_of_inliers : float
        parameter used for the RANSAC. When the ratio of inliers exceeds this value, the iteration will stop.
    Return
    ------
    F : ndarray 3x3, dtype = float
        output fundamental matrix.
    mask : ndarray (N,), dtype = int
        output array of N elements, every element of which is set to 0 for outliers and to 1 for the other points. 
        The array is computed only in the RANSAC methods. For other methods, it is set to all 1’s.
    """

    st0 = np.random.get_state()

    # RANSAC + 8-Point
    if method == cv2.FM_8POINT + cv2.RANSAC or method == cv2.RANSAC:
        np.random.seed(0)
        __, F, mask = m_ransac(find_fundamental_matrix, compute_fundamental_matrix_error, 
            np.hstack([pts1, pts2]), num_samples = 8, thresh = thresh, ratio_of_inliers = ratio_of_inliers)
    # 8-Point only
    elif method == cv2.FM_8POINT:
        np.random.seed(0)
        __, F, __ = m_ransac(find_fundamental_matrix, compute_fundamental_matrix_error, 
            np.hstack([pts1, pts2]), num_samples = len(pts1), max_iter = 1)   
        mask = np.ones(len(pts1))    
    else:
        raise ValueError("We only support 8-point algorithm and RANSAC.") 

    np.random.set_state(st0)
    
    return F, mask

""" ---------------------------------- PnP Algorithm ----------------------------------------- """

def m_solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, thresh = 10000.0, ratio_of_inliers = 0.99):
    """The function estimates an object pose given a set of object points, their corresponding image projections, 
    as well as the camera intrinsic matrix and the distortion coefficients. This function finds such a pose that 
    minimizes reprojection error, that is, the sum of squared distances between the observed projections imagePoints 
    and the projected (using cv2.projectPoints()) objectPoints. The use of RANSAC makes the function resistant to outliers. 

    Parameters
    ----------
    objectPoints : ndarray Nx3, dtype = float
        contains object points in the object coordinate space, where N is the number of points.
    imagePoints : ndarray Nx2, dtype = float
        contains corresponding image points, where N is the number of points.
    cameraMatrix : ndarray 3x3, dtype = float
        input camera intrinsic matrix.
    distCoeffs : ndarray (k,), dtype = float
        input vector of distortion coefficients (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]]) of 4, 5, or 8 elements. 
        If the vector is None/empty, the zero distortion coefficients are assumed.
    thresh : float
        parameter used for RANSAC. It is the maximum distance from a point to an epipolar line in pixels, 
        beyond which the point is considered an outlier and is not used for computing the final fundamental 
        matrix.
    ratio_of_inliers : float
        parameter used for the RANSAC. When the ratio of inliers exceeds this value, the iteration will stop.
    Return
    ------
    retval : boolen
        whether we find the pose.
    rvec : ndarray (3,), dtype = float
        output rotation vector.
    tvec : ndarray (3,), dtype = float
        output translation vector.
    inliers : ndarray (M,), dtype = int
        output vector that contains indices of inliers in objectPoints and imagePoints, where M is the 
        number of inliers.
    """

    # TODO
    
    return cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs)
