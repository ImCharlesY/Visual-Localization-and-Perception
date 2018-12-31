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
    intrinsic_matrix = cv2.decomposeProjectionMatrix(projection_matrix)[0]
    # Calculate extrinsic matrix ([R|t]) from camera matrix and projection matrix
    extrinsic_matrix = np.dot(np.linalg.inv(intrinsic_matrix), projection_matrix)
    R, t = extrinsic_matrix[:3, :3], extrinsic_matrix[:, 3]

    # Calculate reprojection error
    imgpoints2 = cv2.projectPoints(objpoints, cv2.Rodrigues(R)[0], t, intrinsic_matrix, distortion_vector)[0]
    # the return array of cv2.projectPoints shapes as (N, 1, 2)
    imgpoints1 = np.asarray([[row.tolist()] for row in imgpoints])
    error = cv2.norm(imgpoints1, imgpoints2, cv2.NORM_L2) / len(imgpoints2)

    return error

def extract_pose_from_projection(projs):

    poses = []
    for proj in projs:
        intrinsic_matrix = cv2.decomposeProjectionMatrix(proj[0])[0]
        extrinsic_matrix = np.dot(np.linalg.inv(intrinsic_matrix), proj[0])
        R, t = extrinsic_matrix[:3, :3], extrinsic_matrix[:, 3]
        rvec = cv2.Rodrigues(R)[0].ravel()
        tvec = t.ravel()
        poses.append(np.hstack([rvec, tvec]))

    return np.asarray(poses)

def calculate_projection_from_pose(poses, intrinsic_matrix):

    projs = []
    for pose in poses:
        rvec = pose[:3]
        tvec = pose[3:]
        R = cv2.Rodrigues(rvec)[0]
        t = tvec.reshape((-1,1))
        extrinsic_matrix = np.hstack([R, t])
        projs.append(np.dot(intrinsic_matrix, extrinsic_matrix))

    return projs


class bundle_adjustment:
    """Based on tutorial on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html"""

    def __init__(self, camera_params, points_3d, points_2d, camera_indices, point_indices, intrinsic_matrix):
        """Parameters:
            camera_params with shape (n_cameras, 6) 
                contains initial estimates of parameters for all cameras. 
                    First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), 
                    next 3 components form a translation vector.

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

        self.intrinsic_matrix = intrinsic_matrix


    def project(self, points, camera_params, intrinsic_matrix):
        """Convert 3-D points to 2-D by projecting onto images."""

        points_proj = np.zeros((points.shape[0], 2), dtype = points.dtype)
        for i, (point, camera_param) in enumerate(zip(points, camera_params), 1):
            points_proj[i - 1] = cv2.projectPoints(np.asarray([point]), 
                camera_param[:3].reshape(-1,1), camera_param[3:6].reshape(-1,1), intrinsic_matrix, None)[0]
   
        return points_proj


    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, intrinsic_matrix):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices], intrinsic_matrix)
        return (points_proj - points_2d).ravel()


    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

        return A


    def extract_params(self, params, n_cameras, n_points):
        """
        Retrieve camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        return camera_params, points_3d


    def optimize(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """

        n_cameras = self.camera_params.shape[0]
        n_points = self.points_3d.shape[0]

        x0 = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))
        f0 = self.fun(x0, n_cameras, n_points, self.camera_indices, self.point_indices, self.points_2d, self.intrinsic_matrix)

        A = self.bundle_adjustment_sparsity(n_cameras, n_points, self.camera_indices, self.point_indices)

        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(n_cameras, n_points, self.camera_indices, self.point_indices, self.points_2d, self.intrinsic_matrix))

        params = self.extract_params(res.x, n_cameras, n_points)

        return params
