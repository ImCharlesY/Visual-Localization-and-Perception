#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : m_ransac
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-01-03
'''

import numpy as np
import cv2

def find_fundamental_matrix(points):

    return cv2.findFundamentalMat(points[:,:2], points[:,2:], cv2.FM_8POINT)[0]


def find_distance_to_fundamental_matrix(mat, points):

    pts1 = np.hstack([points[:,:2], np.ones(len(points)).reshape(-1,1)])
    pts2 = np.hstack([points[:,2:], np.ones(len(points)).reshape(-1,1)])

    return np.dot(np.dot(pts1, mat), pts2.T).diagonal()


def m_ransac(fit_model, validate_model, X, num_samples, thresh = 3, ratio_of_inliers = 0.6):

    best_model = None
    best_mask = []
    best_ratio = 0.0

    max_iter = int(np.log10(1 - ratio_of_inliers) / np.log10(1 - np.power(0.8, 8)) * 1.2)

    # perform RANSAC iterations
    for it in range(max_iter):

        # randomly select samples
        all_indices = np.arange(X.shape[0])
        np.random.shuffle(all_indices)
     
        indices_1 = all_indices[:num_samples]
        indices_2 = all_indices[num_samples:]
     
        sample_points = X[indices_1,:]
     
        # find a model for sample points
        model = fit_model(sample_points)

        if model is None:
            continue
     
        # find distance to the model for all points     
        dist = validate_model(model, X)
        mask = np.zeros(len(X))
        mask[dist < thresh] = 1
     
        # in case a new model is better - cache it
        if np.count_nonzero(mask) / len(X) > best_ratio:
            best_ratio = np.count_nonzero(mask) / len(X)
            best_model = model
            best_mask = mask
     
        # done in case we have enough inliers
        if np.count_nonzero(mask) > len(X) * ratio_of_inliers:
            break

    return best_model is not None, best_model, best_mask


def m_findFundamentalMat(pts1, pts2, method, thresh = 1.0, ratio_of_inliers = 0.99):

    __, F, mask = m_ransac(find_fundamental_matrix, find_distance_to_fundamental_matrix, 
        np.hstack([pts1, pts2]), 8, thresh = thresh, ratio_of_inliers = ratio_of_inliers)
    
    return F, mask