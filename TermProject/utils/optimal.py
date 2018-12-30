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
from scipy.sparse import lil_matrix
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

class bundle_adjustment:
    """Based on tutorial on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html"""

    def __init__(self, camera_params, points_3d, points_2d, camera_indices, point_indices):
        """Parameters:
            camera_params with shape (n_cameras, 9) 
                contains initial estimates of parameters for all cameras. 
                    First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), 
                    next 3 components form a translation vector, 
                    then a focal distance and two distortion parameters.

            points_3d with shape (n_points, 3) 
                contains initial estimates of point coordinates in the world frame.

            camera_ind with shape (n_observations,) 
                contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.

            point_ind with shape (n_observations,) 
                contatins indices of points (from 0 to n_points - 1) involved in each observation.

            points_2d with shape (n_observations, 2) 
                contains measured 2-D coordinates of points projected on images in each observations.
        """
        self.camera_params = camera_params
        self.points_3d = points_3d
        self.points_2d = points_2d

        self.camera_indices = camera_indices
        self.point_indices = point_indices

    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


    def project(self, points, camera_params):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]
        points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        f = camera_params[:, 6]
        k1 = camera_params[:, 7]
        k2 = camera_params[:, 8]
        n = np.sum(points_proj ** 2, axis=1)
        r = 1 + k1 * n + k2 * n ** 2
        points_proj *= (r * f)[:, np.newaxis]
        return points_proj


    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
        points_3d = params[n_cameras * 9:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        return (points_proj - points_2d).ravel()


    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 9 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(9):
            A[2 * i, camera_indices * 9 + s] = 1
            A[2 * i + 1, camera_indices * 9 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

        return A


    def extract_params(self, params, n_cameras, n_points):
        """
        Retrieve camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
        points_3d = params[n_cameras * 9:].reshape((n_points, 3))

        return camera_params, points_3d


    def bundleAdjust(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """

        n_cameras = self.camera_params.shape[0]
        n_points = self.points_3d.shape[0]

        x0 = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))
        f0 = self.fun(x0, n_cameras, n_points, self.cameraindices, self.point_indices, self.points_2d)

        A = self.bundle_adjustment_sparsity(n_cameras, n_points, self.camera_indices, self.point_indices)

        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(n_cameras, n_points, self.camera_indices, self.point_indices, self.points_2d))

        params = self.extract_params(res.x, n_cameras, n_points, self.camera_indices, self.point_indices, self.points_2d)

        return params
