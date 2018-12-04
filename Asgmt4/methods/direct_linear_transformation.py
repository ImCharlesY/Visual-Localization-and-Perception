#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : direct_linear_transformation
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date     		: 2018-12-04
'''

import numpy as np

def direct_linear_transformation(X, y):
	"""Direct Linear Transformation
	Parameters
	----------
	X : ndarray
		Original vectors.
	y : ndarray
		Target vectors after rotation.
	Return
	------
	sol : ndarray
	    The rotation matrix.
	err : float
		The mean square error.
	"""

	def vector_span(x_vector):
		zero_vector = np.zeros(3)
		return np.vstack([np.hstack([x_vector,zero_vector,zero_vector]),
			np.hstack([zero_vector,x_vector,zero_vector]),
			np.hstack([zero_vector,zero_vector,x_vector])])

	def calc_err(X, y, R):
		""" Calculate the standard deviation.
		"""
		err = y - np.dot(R, X)
		return np.trace(np.dot(err.T, err))

	# Solve the equation directly.
	X_stack = np.vstack([vector_span(col) for col in X.T])
	y_stack = np.hstack([col for col in y.T])
	sol = np.dot(np.linalg.inv(np.dot(X_stack.T, X_stack)), np.dot(X_stack.T, y_stack)).reshape((3,3))
	U,__,VT = np.linalg.svd(sol)
	sol = np.dot(U, VT)
	return sol, calc_err(X, y, sol)


if __name__ == '__main__':

	# Load data.
	from scipy import io
	data = io.loadmat('../data/xy.mat')
	X, y = data['x'], data['y']

	sol, err = direct_linear_transformation(X, y)
	print("  Solution      : \r\n{}".format(sol))
	print("  Error         : {}".format(err))
	print('-'*50)	
