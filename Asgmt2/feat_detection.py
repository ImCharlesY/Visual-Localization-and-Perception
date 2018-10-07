#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : Feature Detection
Author          : Charles Young
Python Version  : Python 3.6.1
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date     		: 2018-10-06
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt # require pillow
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import os
import sys
import time

os.chdir(sys.path[0])

images_dir = os.path.join(".", "images", "input")
result_dir = os.path.join(".", "images", "output")



def get_args():
	parser = argparse.ArgumentParser()
	input_file = "test.jpg"
	parser.add_argument('-i', "--input_file", default = input_file, help = "Original images to be annotated.")
	parser.add_argument('-w', "--windowSize", default = 2, type = np.int, help = "Neighborhood size or the size (side length) of the sliding window.")
	parser.add_argument('-k', default = 0.04, type = np.float32, help = "An empirical constant (0.04-0.06).")
	parser.add_argument('-t', "--thresh", default = 0.001, type = np.float32, help = "The threshold above which a corner is counted.")
	return parser.parse_args()



def harrisCorner(src, windowSize, k):
	"""
	This function runs the Harris feature detector on the input image. 
	For each pixel (x,y) it calculates a 2 * 2 gradient covariance matrix
	M(x,y) over a windowSize * windowSize neighborhood. Then it computes 
	the following characteristic:
		dst(x,y) = det(M) - k * (tr(M))^2, where k is an empirical constant (0.04-0.06)

	Arguments:
	-----------
		src: np.ndarray
		    Input image
		windowSize: int
			Neighborhood size or the size (side length) of the sliding window.
			For every pixel p, this function considers a windowSize * windowSize neighborhood S(p).
			It calculates the covariation matrix of derivatives over the neighborhood as: 
			M = [ sum((I_x)^2)   ,  sum((I_x)(I_y));
				  sum((I_y)(I_x)),  sum((I_y)^2)    ]
		k: float
			An empirical constant (0.04-0.06).

	Returns:
	-----------
		dst: np.ndarray
			Image to store the Harris detector responses.
	""" 

	height = src.shape[0]
	width = src.shape[1]

	# Calculate the global auto-correlation matrix. 
	Ix = cv2.Sobel(src, -1, 1, 0) # Ix = cv2.filter2D(src, -1, np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
	Iy = cv2.Sobel(src, -1, 0, 1) # Iy = cv2.filter2D(src, -1, np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))  
	Ixx, Ixy, Iyy = Ix**2, Iy*Ix, Iy**2

	# Image to store the value of the Harris detector response.
	dst = np.zeros(src.shape)

	offset = windowSize // 2
	# Loop through image
	for y in range(offset, height-offset):
		for x in range(offset, width-offset):
			# Calculate the local matrix M.
			# M = [ sum((I_x)^2)   ,  sum((I_x)(I_y));
			#       sum((I_y)(I_x)),  sum((I_y)^2)    ]
			#   = [ Sxx, Sxy;
			#       Syx, Syy ]
			Sxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1].sum()
			Sxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1].sum()
			Syy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1].sum()

			# Compute determinant, trace, and corner response.
			det = (Sxx * Syy) - (Sxy**2)
			trace = Sxx + Syy
			r = det - k*(trace**2)
			dst[y, x] = r

	return dst



def findLocalMax(ridx, dst):
	"""
	This function runs the Harris feature detector on the input image. 
	For each pixel (x,y) it calculates a 2 * 2 gradient covariance matrix
	M(x,y) over a windowSize * windowSize neighborhood. The it computes 
	the following characteristic:
		dst(x,y) = det(M) - k * (tr(M))^2, where k is an empirical constant (0.04-0.06)

	Arguments:
	-----------
		ridx: np.ndarray, dtype = bool
		    Boolen matrix to store where in dst the reponse value is larger than threshold.
		dst: np.ndarray
		    Image to store the Harris detector responses.

	Returns:
	-----------
		cidx: np.ndarray, dtype = bool
		    Boolen matrix to store where in dst the reponse value is larger than threshold and also the maximum in local region.
	"""	

	'''Note:
	Non-local-maxima Suppression:
	In order to detect all the local maxima in the image intensity regardless of their intensity value. We first apply a dilation 
	to the orginal corner responses, then find the pixels where the original image and its dilated version have the same value. 
	This works because, by definition, dilation(x, y, E, dst) = max_{(x,y) in E} (dst_{x,y}), and therefore 
	dilation(x, y, E, dst) = dst(x, y) whenever (x,y) is the location of a local maximum at the scale of E.
	'''

	cidx = np.zeros(dst.shape).astype('bool')
	# Here we apply a dilation to dst and find the pixels whose values keep still during dilation
	# then choose pixels that are also in ridx.
	cidx[ridx & (dst == cv2.dilate(dst, None))] = True 
	corner_list = np.argwhere(cidx == True)
	for corner in corner_list:
		cidx[corner[0]-1:corner[0]+2, corner[1]-1:corner[1]+2] = True

	return cidx



def disp(img, args):
	# Unpack all images.
	src, dst, region, corners, marked = img
	# Normalize dst.
	dmin, dmax = dst.min(), dst.max()
	dst = (dst - dmin) / (dmax - dmin)

	# Prepare display.
	fig = plt.figure(figsize = (19.2,10.8))
	gs = plt.GridSpec(2, 4)
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[0, 1], sharex = ax1, sharey = ax1)
	ax3 = fig.add_subplot(gs[1, 0], sharex = ax2, sharey = ax2)
	ax4 = fig.add_subplot(gs[1, 1], sharex = ax3, sharey = ax3)
	ax5 = fig.add_subplot(gs[0:2, 2:4])
	for aa in (ax1, ax2, ax3, ax4, ax5):
		aa.set_axis_off()

	ax1.imshow(src)
	ax1.set_title('Original Image')

	im_dst = ax2.matshow(dst, cmap = 'hot')
	ax2.set_title('Dst')
	divider = make_axes_locatable(ax2)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(im_dst, cax=cax, orientation='vertical');

	ax3.imshow(region, cmap = plt.cm.gray)
	ax3.set_title('Region')

	ax4.imshow(corners, cmap = plt.cm.gray)
	ax4.set_title('Corners')

	ax5.imshow(marked)
	ax5.set_title('Marked')

	# Save graph to file.
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	result_file = args.input_file.split('.')[0] + '_annotated_w' + str(args.windowSize) + 'k' + str(args.k) + 't' + str(args.thresh) + '.svg'
	plt.savefig(os.path.join(result_dir, result_file), bbox_inches = 'tight', format = 'svg')
	plt.show()



def main():
	# Get command line arguments.
	args = get_args()
	src = cv2.imread(os.path.join(images_dir, args.input_file))
	gray = np.float32(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY))
	print('Input Image: {}, windowSize = {}, k = {}, threshold = {}'.format(gray.shape, args.windowSize, args.k, args.thresh))

	start_time = time.time()
	# Step 1: Compute corner response.
	# dst = cv2.cornerHarris(gray,args.windowSize,3,args.k)
	dst = harrisCorner(gray, args.windowSize, args.k)     
	print("--- Step 1: Compute corner response: %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	# Step 2: Find region R > threshold.
	ridx = dst > args.thresh * dst.max()
	region = np.zeros(gray.shape).astype('int')
	region[ridx] = 255
	print("--- Step 2: Find region R > threshold: %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	# Step 3: Find local maximum in R.
	cidx = findLocalMax(ridx, dst)
	print("--- Step 3: Find local maximum in R: %s seconds ---" % (time.time() - start_time))	

	start_time = time.time()
	# Step 4:  Mark corners.
	corners = np.zeros(gray.shape).astype('int')
	corners[cidx] = 255
	marked = src.copy()
	marked[cidx] = [255,0,0]
	print("--- Step 4: Mark corners: %s seconds ---" % (time.time() - start_time))	

	disp((src, dst, region, corners, marked), args)


if __name__ == '__main__':
	main()
