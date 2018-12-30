#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : visualize
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-12-30
'''

import os
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def drawLines(img1, img2, lines, pts1, pts2, colors):

    r,c = img1.shape
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for i, (r, pt1, pt2, color) in enumerate(zip(lines, pts1, pts2, colors), 1):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img1 = cv2.putText(img1, '{}'.format(i), tuple(pt1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def drawEpilines(img1, img2, pts1, pts2, F, colors):

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    new_img1, __ = drawLines(img1, img2, lines1, pts1, pts2, colors)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    new_img2, __ = drawLines(img2, img1, lines2, pts2, pts1, colors)

    return new_img1, new_img2

def scatter3DPoints(pts, colors, output_path = None):

    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize = 16)
    ax = fig.gca(projection = '3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')

    for i, (pt, c) in enumerate(zip(pts, colors), 1):
        ax.scatter(pt[0], pt[1], pt[2], c = np.asarray([c]) / 255, marker = 'o')
        ax.text(pt[0], pt[1], pt[2], '{}'.format(i), size = 10, zorder = 1, color = np.asarray(c) / 255)
        
    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, '3DPointsCloud.svg'), bbox_inches = 'tight', format = 'svg')