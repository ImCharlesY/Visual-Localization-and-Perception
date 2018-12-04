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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from methods import * # icp

def calc_err(A, B, T):

    src = np.ones((4,A.shape[0]))
    dst = np.ones((4,B.shape[0]))
    src[:3,:] = np.copy(A.T)
    dst[:3,:] = np.copy(B.T)
    dst_m = np.dot(T, src)
    err = dst_m - dst
    return np.trace(np.dot(err.T, err))

def display(x, y, y_transformed):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[0,:], x[1,:], x[2,:], c='r', marker='o')
    ax.scatter(y[0,:], y[1,:], y[2,:], c='y', marker='*')
    ax.scatter(y_transformed[0,:], y_transformed[1,:], y_transformed[2,:], c='b', marker='+')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.legend(['source points cloud (src)', 'destination points cloud (dst)', 'src after transformation'])

def solve():

    # Load data.
    from scipy import io
    data = io.loadmat('./data/xy.mat')
    x, y = data['x'], data['y']

    T, __ = icp(x.T, y.T)

    print('The rigid body transformation: \n{0}'.format(T))

    src = np.ones((4, x.shape[1]))
    src[:3,:] = np.copy(x)
    y_transformed = np.dot(T, src)[:3,]

    display(x, y, y_transformed)    

    if not os.path.exists('./result/'):
        os.makedirs('./result/')    
    io.savemat('./result/solution.mat', {'T':T})    

def validate():

    # Load data.
    from scipy import io
    data = io.loadmat('./result/solution.mat')
    T = data['T']   
    
    x = np.random.uniform(0,10,(3,1000))
    src = np.ones((4, 1000))
    src[:3,:] = np.copy(x)
    y = np.dot(T, src)[:3,]

    T_transformed, __ = icp(x.T, y.T)

    src = np.ones((4, x.shape[1]))
    src[:3,:] = np.copy(x)
    y_transformed = np.dot(T_transformed, src)[:3,]

    print('Rigid body transformation: \n{0}'.format(T_transformed))
    print('MSE between destination points cloud and source points cloud after transformation: {0}'.format(calc_err(x.T,y.T,T_transformed)))

    display(x, y, y_transformed)

if __name__ == '__main__':

    np.random.seed(0)
    solve()
    validate()
    plt.show()
