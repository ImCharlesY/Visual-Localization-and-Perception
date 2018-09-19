#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : HistogramSpecialization
Author          : Charles Young
Python Version  : Python 3.5.4
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date     		: 2018-09-18
'''

'''This script implements a histogram specialization algorithm described as follows.
Histogram Specialization Algorithm (HSA):
Step 1: Calculate the cumulative histogram of the first image: f_1(I).
Step 2: Calculate the cumulative histogram of the second image: f_2(I).
Step 3: Build a lookup table I' = finv_2(f_1(I)) by finding the corresponding gray level
	I'_j of each gray level I_i, where I'_j = argmin_j |f_1(I_i) - f_2(I'_j)|.
Step 4: Map the new intensity of each pixel by finding the lookup table.
'''

import numpy as np
import matplotlib.pyplot as plt # require pillow
import matplotlib.image as mpimg
import argparse
import os.path

images_dir = "images/input/"

def get_args():
    parser = argparse.ArgumentParser()
    origin_file = "test_o.jpg"
    target_file = "test_t.jpg"
    parser.add_argument('-o', "--origin_file", default = origin_file, help = "Original images to be specialized.")
    parser.add_argument('-t', "--target_file", default = target_file, help = "Target images.")
    return parser.parse_args()


def rgb2gray(rgb_image):
	"""
	Transform the input rgb image to corresponding grayscale image.grayscale

	Arguments:
    -----------
        rgb_image: np.ndarray
            Image to transform

    Returns:
    -----------
        gray_image: np.ndarray
            The transformed output image	
	"""
	return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])


def imhist(img):
	"""
	Get the histogram of input image.

    Arguments:
    -----------
        img: np.ndarray
            Input image

    Returns:
    	pixel_val: np.ndarray
			The set of unique pixel values, s.t. the x axis values of histogram.
		pixel_cnt: np.ndarray
			Corresponding counts of the pixel_val, s.t. the y axis values of histogram. 
    -----------

	"""	
	pixel_val, pixel_cnt = np.unique(img, return_counts = True)
	return pixel_val, pixel_cnt


def ecdf(img):
	"""
	Compute the empirical cumulative distribution function of the input image.

    Arguments:
    -----------
        img: np.ndarray
            Input image

    Returns:
    -----------
    	pixel_val: np.ndarray
			The set of unique pixel values, s.t. the x axis values of the empirical cumulative distribution.
		ecdf: np.ndarray
			The cumulative probability density values, s.t. the y axis values of histogram. 
    """

	# Get the set of unique pixel values and their corresponding counts.
	pixel_val, pixel_cnt = imhist(img)
	# Calculate cumsum.
	ecdf = np.cumsum(pixel_cnt).astype(np.float64)
	# Normalize to [0,1].
	ecdf /= ecdf[-1]
	return pixel_val, ecdf


def hist_match(x_org, y_org, x_tar, y_tar):
    """
    Build the lookup table described in step 3.

    Arguments:
    -----------
    	x_org, y_org: np.ndarray
    		The cumulative distribution function of orginal image.
    	x_tar, y_tar: np.ndarray
    		The cumulative distribution function of target image.

    Returns:
    -----------
    	x_lookup, y_lookup: np.ndarray
    		Map from original brightness to target brightness.
    """
    x_lookup = np.zeros(x_org.shape)
    y_lookup = np.zeros(y_org.shape)
    
    for i in range(x_lookup.shape[0]):
    	x_lookup[i] = x_org[i]
    	y_lookup[i] = x_tar[np.argmin(np.abs(y_tar - y_org[i]))]

    return x_lookup, y_lookup


def hist_specialize(img_org, img_tar, x_lookup, y_lookup):
	"""
    Map the new intensity of each pixel by finding the lookup table.

    Arguments:
    -----------
		img_org: np.ndarray
			The original image.
		img_tar: np.ndarray
			The target image.
    	x_lookup, y_lookup: np.ndarray
    		Map from original brightness to target brightness.

    Returns:
    	img_adj: np.ndarray
    		The adjusted image.
    -----------	
	
	"""
	img_adj = np.zeros(img_org.shape)
	row, col = img_org.shape
	for i in range(row):
		for j in range(col):
			org_idx = np.argwhere(x_lookup == img_org[i][j])[0][0]
			img_adj[i][j] = y_lookup[org_idx]
	return img_adj


def disp(img):
	img_org, img_adj, img_tar = img

	x_org_hist, y_org_hist = imhist(img_org)
	x_adj_hist, y_adj_hist = imhist(img_adj)
	x_tar_hist, y_tar_hist = imhist(img_tar)	

	x_org_cdf, y_org_cdf = ecdf(img_org)
	x_adj_cdf, y_adj_cdf = ecdf(img_adj)
	x_tar_cdf, y_tar_cdf = ecdf(img_tar)		

 	# Plot result.
	fig = plt.figure()
	gs = plt.GridSpec(3, 3)
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[0, 1], sharex = ax1, sharey = ax1)
	ax3 = fig.add_subplot(gs[0, 2], sharex = ax1, sharey = ax1)
	ax4 = fig.add_subplot(gs[1, 0])
	ax5 = fig.add_subplot(gs[1, 1], sharex = ax4, sharey = ax4)
	ax6 = fig.add_subplot(gs[1, 2], sharex = ax4, sharey = ax4)
	ax7 = fig.add_subplot(gs[2, :])
	for aa in (ax1, ax2, ax3):
	    aa.set_axis_off()

	ax1.imshow(img_org, cmap = plt.cm.gray)
	ax1.set_title('Original')
	ax2.imshow(img_adj, cmap = plt.cm.gray)
	ax2.set_title('Adjusted')
	ax3.imshow(img_tar, cmap = plt.cm.gray)
	ax3.set_title('Target')

	ax4.plot(x_org_hist, y_org_hist, '--k', lw = 1, label = 'Original')
	ax4.fill_between(x_org_hist, y_org_hist, interpolate = True, color = 'green', alpha = 0.5)
	ax5.plot(x_adj_hist, y_adj_hist, '--k', lw = 1, label = 'Adjusted')
	ax5.fill_between(x_adj_hist, y_adj_hist, interpolate = True, color = 'green', alpha = 0.5)
	ax6.plot(x_tar_hist, y_tar_hist, '--k', lw = 1, label = 'Target')
	ax6.fill_between(x_tar_hist, y_tar_hist, interpolate = True, color = 'green', alpha = 0.5)

	ax7.plot(x_org_cdf, y_org_cdf, '-r', lw = 3, label = 'Original')
	ax7.plot(x_adj_cdf, y_adj_cdf, '-b', lw = 3, label = 'Adjusted')
	ax7.plot(x_tar_cdf, y_tar_cdf, '--y', lw = 3, label = 'Target')
	ax7.set_xlim(x_org_cdf[0], x_org_cdf[-1])
	ax7.set_xlabel('Pixel value')
	ax7.set_ylabel('Cumulative %')
	ax7.legend(loc = 5)	

	plt.show()


def main():
	# Get input images path.
	args = get_args()

	# Read in input images.
	img_org = rgb2gray(mpimg.imread(os.path.join(images_dir, args.origin_file)))
	img_tar = rgb2gray(mpimg.imread(os.path.join(images_dir, args.target_file)))

	# Implement step 1 and step 2 -- calculate the cumulative histograms of both images.
	x_org_cdf, y_org_cdf = ecdf(img_org)
	x_tar_cdf, y_tar_cdf = ecdf(img_tar)

	# Implement step 3 -- calculate lookup table.
	x_lookup, y_lookup = hist_match(x_org_cdf, y_org_cdf, x_tar_cdf, y_tar_cdf)

	# Implement step 4 -- map the new intensity of each pixel.
	img_adj = hist_specialize(img_org, img_tar, x_lookup, y_lookup)

	# Display results.
	disp((img_org, img_adj, img_tar))


if __name__ == '__main__':
	main()
