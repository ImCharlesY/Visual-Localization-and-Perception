#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : gauss_newton
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date     		: 2018-12-04
'''

import numpy as np

def gauss_newton(X, y, R0, tol = 1e-10, maxits = 256):
	"""Gauss-Newton algorithm.
	Parameters
	----------
	X : ndarray
		Original vectors.
	y : ndarray
		Target vectors after rotation.
	R0 : ndarray
		Initial guesses or starting estimates.  
	tol : float
	    Tolerance threshold. The problem is considered solved when this value
	    becomes larger than the change of the correction vector.
	    Defaults to 1e-10.
	maxits : int
	    Maximum number of iterations of the algorithm to perform.
	    Defaults to 256.
	Return
	------
	Rn : ndarray
	    Resultant values. In this problem, the rotation matrix.
	err : float
		The standard deviation.
	its : int
	    Number of iterations performed.
	errs : list
		The standard deviations during iteration.
	"""

	dx = np.ones(len(R0))   # Correction vector
	Rn = np.array(R0)       # Approximation of solution
	errs = []

	def skew(u):
		# Span a (3,1) vector into its corresponding (3,3) skew symmetric matrix.

		return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

	def calc_step(X, y, Rn):
		# Solve function dx = inv(J'J)J'z.

		# Calculate Jacobian matrix.
		p = np.dot(Rn, X)
		# For independent errors the whole Jacobian is just the stacked matrix of individual Jacobians.
		J = np.vstack([-skew(pi) for pi in p.T])

		# Calculate error.
		err = y - p
		# For independent errors the whole error is just the stacked vector of individual errors.
		z = np.hstack([col for col in (y - p).T])

		return np.dot(np.linalg.inv(np.dot(J.T, J)), np.dot(J.T, z))

	def calc_err(X, y, R):
		# Calculate the standard deviation.
		err = y - np.dot(R, X)
		return np.trace(np.dot(err.T, err))

	its = 0
	while (its < maxits) and (dx[np.abs(dx) > tol].size > 0):
		# Calculate correction vector.
		dx = calc_step(X, y, Rn)
		# Update R.
		Rn = Rn + np.dot(skew(dx), Rn) # Adjust R.
		U,__,VT = np.linalg.svd(Rn)    # Apply SVD decomposition.
		Rn = np.dot(U, VT)             # Obtain the nearest rotation matrix.

		assert (Rn.T - np.linalg.inv(Rn) < 1e-10).all()	# Check if orthogonal.
		assert np.linalg.det(Rn) - 1 < 1e-10            # Check if its det is 1.

		# Update iteration.
		its += 1

		err = calc_err(X, y, Rn)
		errs.append(err)
		# print('  Iter: {0}, Error: {1}'.format(its, calc_err(X, y, Rn)))

	return Rn, err, its, errs


if __name__ == '__main__':

	# Load data.
	from scipy import io
	data = io.loadmat('./data/xy.mat')
	X, y = data['x'], data['y']

	U,__,VT = np.linalg.svd(np.zeros((3,3)))
	R0 = np.dot(U, VT)
	sol, err, __, __ = gauss_newton(X, y, R0)
	print("  Solution      : \r\n{}".format(sol))
	print("  Error         : {}".format(err))
	print('-'*50)
