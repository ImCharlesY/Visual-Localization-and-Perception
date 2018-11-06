#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : Gauss_Newton
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date     		: 2018-11-05
'''

import numpy as np
import os
import sys
import argparse
from scipy import io
import matplotlib.pyplot as plt

# If run in Windows, comment this line.
# plt.switch_backend('agg')

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--maxits', default = 256, type = np.int, nargs = '?', help = "Maximum number of iterations of the algorithm to perform. Default 256.")
	parser.add_argument('--seed', dest = 'random_seed', default = 0, type = np.int, nargs = '?', help = "Random seed. Default 0")
	parser.add_argument('--num_init', default = 10, type = np.int, nargs = '?', help = "Number of initial guesses. Default 10.")
	parser.add_argument('--orth', dest = 'orthogonal', action = 'store_const', const = True, help = "Whether to orthogonalize the initial guesses.")
	parser.add_argument('--result', dest = 'save_result', action = 'store_const', const = True, help = "Whether to save the results.")
	parser.add_argument('--legend', dest = 'show_legend', action = 'store_const', const = True, help = "Whether to show the legend in error graph.")
	parser.add_argument('--plot', dest = 'show_graph', action = 'store_const', const = True, help = "Whether to show the error graph.")
	return parser.parse_args()

def calc_err(X, y, R):
	# Calculate the standard deviation.
	err = y - np.dot(R, X)
	return np.trace(np.dot(err.T, err))

def solve(X, y, R0, tol = 1e-10, maxits = 256):
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

	args = get_args()

	np.random.seed(args.random_seed)

	# Load data.
	data = io.loadmat('./data/xy.mat')
	X, y = data['x'], data['y']

	# Solve the equation directly.
	sol = np.dot(y, np.linalg.inv(X))
	err = calc_err(X, y, sol)
	print("  Solution      : \r\n{}".format(sol))
	print("  Error         : {}".format(err))
	print('-'*50)	
	if args.save_result:		
		io.savemat('./result/solution0.mat', {'R':sol})

	# Generate some initial guesses.
	starts = []
	for i in range(args.num_init):
		R = np.random.randn(9).reshape((3,3)).astype('float64')
		if args.orthogonal:
			U,__,VT = np.linalg.svd(R)
			R = np.dot(U, VT)
		starts.append(R)

	if not os.path.exists('./result/'):
		os.makedirs('./result/')

	# Perform Gauss-Newton algprithm on each initial guess.
	errs_all = []
	min_err = err
	best_sol = sol
	for i, R0 in enumerate(starts):
		print("Start {}:".format(i + 1))
		print("  Initial guess : \r\n{}".format(R0))
		sol, err, its, errs = solve(X, y, R0, maxits = args.maxits)
		print("  Iterations    : {}".format(its))
		print("  Solution      : \r\n{}".format(sol))
		print("  Error         : {}".format(err))
		print('-'*50)
		errs_all.append(errs)
		if err < min_err:
			min_err = err
			best_sol = sol

	print("  Best Solution : \r\n{}".format(best_sol))
	print("  Error         : {}".format(min_err))
	print('-'*50)	
	if args.save_result:		
		io.savemat('./result/solution_best.mat', {'R':best_sol})			

	# Display
	fig = plt.figure(figsize = (19.2,10.8))
	plt.title('Error')
	for i in range(len(starts)):
		plt.plot(np.arange(len(errs_all[i])), errs_all[i], label = 'Guess '+ str(i + 1))
	if args.show_legend:
		plt.legend()

	# Save graph to file.
	plt.savefig('./result/error' + str(len(starts)) + '{}'.format(args.orthogonal) + '.svg', bbox_inches = 'tight', format = 'svg')	
	if args.show_graph:
		plt.show()
