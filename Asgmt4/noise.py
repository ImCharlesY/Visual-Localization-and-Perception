#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : noise
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-12-04
'''

import numpy as np
from methods import * # direct_linear_transformation, absolute_orientation, gauss_newton
import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(20)

    # Load data.
    from scipy import io
    data = io.loadmat('./data/xy.mat')
    X, y = data['x'], data['y']

    # Generate initial guess for Gauss_Newton Algorithm
    U,__,VT = np.linalg.svd(np.random.randn(9).reshape((3,3)).astype('float64'))
    R0 = np.dot(U, VT)

    __, err_dlt = direct_linear_transformation(X, y)
    __, __, err_quat = absolute_orientation(X, y)
    __, err_gn, its, __ = gauss_newton(X, y, R0)

    # List to record errors and iterations
    errs_ls = np.array([err_dlt, err_quat, err_gn]).reshape((3,1))
    its_ls = np.array([its])

    vars = np.linspace(0,100,100)
    for var in vars:
        X_tmp, y_tmp = X.copy(), y.copy()

        noise = np.random.uniform(0, var, (3,1))
        X_tmp += noise
        y_tmp += noise

        __, err_dlt = direct_linear_transformation(X_tmp, y_tmp)
        __, __, err_quat = absolute_orientation(X_tmp, y_tmp)
        __, err_gn, its, __ = gauss_newton(X_tmp, y_tmp, R0)

        errs_ls = np.hstack([errs_ls, np.array([err_dlt, err_quat, err_gn]).reshape((3,1))])
        its_ls = np.hstack([its_ls, its])

    fig = plt.figure()
    plt.title('Influence of noise')
    ax1 = fig.add_subplot(111)
    for i in range(3):
        ax1.plot(np.hstack([0, vars]), errs_ls[i])
    ax1.set_xlabel('variance')
    ax1.set_ylabel('residual')
    ax1.legend(['Direct Linear Transformation', 'Absolute Orientation', 'Gauss-Newton Algorithm'])

    ax2 = ax1.twinx()
    ax2.bar(np.hstack([0, vars]), its_ls, facecolor = 'y', alpha = 0.3)
    ax2.set_ylim([0, 200])
    ax2.set_ylabel('iterations')
    ax2.legend(['Iterations of Gauss-Newton Algorithm'])
    plt.show()
