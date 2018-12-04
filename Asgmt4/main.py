#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : main
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date     		: 2018-12-04
'''

import os
import numpy as np
import argparse
from methods import * # direct_linear_transformation, absolute_orientation, gauss_newton

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--maxits', default = 256, type = np.int, nargs = '?', help = "Maximum number of iterations of the algorithm to perform. Default 256.")
	parser.add_argument('--seed', dest = 'random_seed', default = 0, type = np.int, nargs = '?', help = "Random seed. Default 0")
	parser.add_argument('--custom_data', dest = 'custom', action = 'store_const', const = True, help = "Whether to use custom dataset.")
    parser.add_argument('--save', dest = 'save_result', action = 'store_const', const = True, help = "Whether to save the results.")
	return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    np.random.seed(args.random_seed)

    # Load data.
    if args.custom:
        X = np.random.uniform(0,10,(3,3))
        U,__,VT = np.linalg.svd(np.random.randn(9).reshape((3,3)).astype('float64'))
        R = np.dot(U, VT)
        y = np.dot(R, X)
        print('Rotation matrix to be solve: \n{0}'.format(R))
    else:
        from scipy import io
        data = io.loadmat('./data/xy.mat')
        X, y = data['x'], data['y']

    # --------------------------------------------------------------------------------------------
    print('==> Solve via direct linear transformation (closed form method)..')
    sol_dlt, err_dlt = direct_linear_transformation(X, y)
    print("  Solution      : \r\n{}".format(sol_dlt))
    print("  Error         : {}".format(err_dlt))
    print('-'*50)	

    # --------------------------------------------------------------------------------------------
    print('==> Solve via absolute orientation (closed form method)..')
    __, sol_quat, err_quat = absolute_orientation(X, y)
    print("  Solution      : \r\n{}".format(sol_quat))
    print("  Error         : {}".format(err_quat))
    print('-'*50)

    # --------------------------------------------------------------------------------------------
    print('==> Solve via gauss-newton method..')
    # Generate initial guess.
    U,__,VT = np.linalg.svd(np.random.randn(9).reshape((3,3)).astype('float64'))
    R0 = np.dot(U, VT)
    print("  Initial guess : \r\n{}".format(R0))
    sol_gn, err_gn, its, __ = gauss_newton(X, y, R0, maxits = args.maxits)
    print("  Iterations    : {}".format(its))
    print("  Solution      : \r\n{}".format(sol_gn))
    print("  Error         : {}".format(err_gn))
    print('-'*50)

	# --------------------------------------------------------------------------------------------
    if args.save_result:
        if not os.path.exists('./result/'):
            os.makedirs('./result/')	
        io.savemat('./result/solution.mat', {'R_dlt':sol_dlt, 'R_quat':sol_quat, 'R_gn_init':R0, 'R_gn':sol_gn})
