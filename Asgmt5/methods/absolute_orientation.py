#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : absolute_orientation
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date     		: 2018-12-04
'''

import math
import numpy as np

def absolute_orientation(X, y):

	def skew_right(u):
		"""Span a (3,1) vector or a (4,1) quaternion into its corresponding (4,4) right multipy skew symmetric matrix.
		"""

		if u.size == 3:
			return np.array([
				[   0, -u[0], -u[1], -u[2]], 
				[u[0],     0,  u[2], -u[1]], 
				[u[1], -u[2],     0,  u[0]], 
				[u[2],  u[1], -u[0],    0]])
		elif u.size == 4:
			return np.array([
				[u[0], -u[1], -u[2], -u[3]], 
				[u[1],  u[0],  u[3], -u[2]], 
				[u[2], -u[3],  u[0],  u[1]], 
				[u[3],  u[2], -u[1],  u[0]]])
		else:
			return np.identity(4)

	def skew_left(u):
		"""Span a (3,1) vector or a (4,1) quaternion into its corresponding (4,4) left multipy skew symmetric matrix.
		"""

		if u.size == 3:
			return np.array([
				[   0, -u[0], -u[1], -u[2]], 
				[u[0],     0, -u[2],  u[1]], 
				[u[1],  u[2],     0, -u[0]], 
				[u[2], -u[1],  u[0],    0]])
		elif u.size == 4:
			return np.array([
				[u[0], -u[1], -u[2], -u[3]], 
				[u[1],  u[0], -u[3],  u[2]], 
				[u[2],  u[3],  u[0], -u[1]], 
				[u[3], -u[2],  u[1],  u[0]]])
		else:
			return np.identity(4)

	def quaternion_matrix(quaternion):
	    """Return homogeneous rotation matrix from quaternion.
	    """

	    q = np.array(quaternion, dtype=np.float64, copy=True)
	    n = np.dot(q, q)
	    q *= math.sqrt(2.0 / n)
	    q = np.outer(q, q)
	    return np.array([
	        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
	        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
	        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])

	def calc_err(X, y, R):
		# Calculate the standard deviation.
		err = y - np.dot(R, X)
		return np.trace(np.dot(err.T, err))

	A = np.sum([np.dot(skew_left(y).T, skew_right(x)) for x, y in zip(X.T, y.T)], axis=0)
	eigenvalues, eigenvectors = np.linalg.eig(A)

	assert (eigenvectors.T - np.linalg.inv(eigenvectors) < 1e-10).all() # Check if orthogonal.
	assert np.linalg.det(eigenvectors) - 1 < 1e-10                      # Check if its det is 1.

	quat = eigenvectors[:,np.argmax(eigenvalues)]
	rotation_matirx = quaternion_matrix(quat)
	return quat, rotation_matirx, calc_err(X, y, rotation_matirx)


if __name__ == '__main__':

	# Load data.
	from scipy import io
	data = io.loadmat('../data/xy.mat')
	X, y = data['x'], data['y']

	__, sol, err = absolute_orientation(X, y)
	print("  Solution      : \r\n{}".format(sol))
	print("  Error         : {}".format(err))
	print('-'*50)	
