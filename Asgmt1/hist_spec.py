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
import os
import sys
import time

os.chdir(sys.path[0])

images_dir = os.path.join(".", "images", "input")
result_dir = os.path.join(".", "images", "output")

def get_args():
    parser = argparse.ArgumentParser()
    origin_file = "test_o.jpg"
    target_file = "test_t.jpg"
    result_file = "test.svg"
    parser.add_argument('-o', "--origin_file", default = origin_file, help = "Original images to be specialized.")
    parser.add_argument('-t', "--target_file", default = target_file, help = "Target images.")
    parser.add_argument('-r', "--result_file", default = result_file, help = "File name to save result graph.")
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
    -----------
    	hist: dict
    		keys: The set of unique pixel values, s.t. the x axis values of histogram.
    		values: Corresponding counts of the pixel_val, s.t. the y axis values of histogram. 			
	"""	

	pixel_val, pixel_cnt = np.unique(img, return_counts = True)
	return dict(zip(pixel_val, pixel_cnt))


def ecdf(img):
	"""
	Compute the empirical cumulative distribution function of the input image.

    Arguments:
    -----------
        img: np.ndarray
            Input image

    Returns:
    -----------
    	cdf: dict
    		keys: The set of unique pixel values, s.t. the x axis values of the empirical cumulative distribution.
    		values: The cumulative probability density values of corresponding keys, s.t. the y axis values of histogram. 			
    """

	# Get the set of unique pixel values and their corresponding counts.
	hist = imhist(img)
	# Calculate cumsum.
	pixel_ecdf = np.cumsum(list(hist.values())).astype(np.float64)
	# Normalize to [0,1].
	pixel_ecdf /= pixel_ecdf[-1]
	return dict(zip(list(hist.keys()), pixel_ecdf))


def hist_match(cdf_org, cdf_tar):
    """
    Build the lookup table described in step 3.

    Arguments:
    -----------
    	cdf_org: dict
    		The cumulative distribution function of orginal image.    		
		cdf_tar: dict
    		The cumulative distribution function of target image.

    Returns:
    -----------
    	lktbl: dict
    		Map from original brightness to target brightness.
    """

    lktbl = dict()
    x_cdf_tar = list(cdf_tar.keys())
    y_cdf_tar = np.array(list(cdf_tar.values()))
    for x, y in cdf_org.items():
    	lktbl[x] = x_cdf_tar[np.argmin(abs(y_cdf_tar - y))]
    return lktbl


def hist_specialize(img_org, lktbl):
	"""
    Map the new intensity of each pixel by finding the lookup table.

    Arguments:
    -----------
		img_org: np.ndarray
			The original image.
    	lktbl: dict
    		Map from original brightness to target brightness.

    Returns:
    	img_adj: np.ndarray
    		The adjusted image.
    """
    
	return np.vectorize(lktbl.__getitem__)(img_org)


def disp(img, args):
	img_org, img_adj, img_tar = img

	hist_org = imhist(img_org)	
	hist_adj = imhist(img_adj)
	hist_tar = imhist(img_tar)

	cdf_org = ecdf(img_org)
	cdf_adj = ecdf(img_adj)
	cdf_tar = ecdf(img_tar)

 	# Plot result.
	fig = plt.figure(figsize = (19.2,10.8))
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

	ax4.plot(list(hist_org.keys()), list(hist_org.values()), '--k', lw = 1, label = 'Original')
	ax4.fill_between(list(hist_org.keys()), list(hist_org.values()), interpolate = True, color = 'green', alpha = 0.5)
	ax5.plot(list(hist_adj.keys()), list(hist_adj.values()), '--k', lw = 1, label = 'Adjusted')
	ax5.fill_between(list(hist_adj.keys()), list(hist_adj.values()), interpolate = True, color = 'green', alpha = 0.5)
	ax6.plot(list(hist_tar.keys()), list(hist_tar.values()), '--k', lw = 1, label = 'Target')
	ax6.fill_between(list(hist_tar.keys()), list(hist_tar.values()), interpolate = True, color = 'green', alpha = 0.5)

	ax7.plot(list(cdf_org.keys()), list(cdf_org.values()), '-r', lw = 3, label = 'Original')
	ax7.plot(list(cdf_adj.keys()), list(cdf_adj.values()), '-b', lw = 3, label = 'Adjusted')
	ax7.plot(list(cdf_tar.keys()), list(cdf_tar.values()), '--y', lw = 3, label = 'Target')
	ax7.set_xlim(0, 255)
	ax7.set_xlabel('Pixel value')
	ax7.set_ylabel('Cumulative %')
	ax7.legend(loc = 5)	

	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	plt.savefig(os.path.join(result_dir, args.result_file), bbox_inches = 'tight', format = args.result_file.split('.')[-1])
	plt.show()



def main():
	start_time = time.time()
	# Get input images path.
	args = get_args()
	# Read in input images.
	img_org = rgb2gray(mpimg.imread(os.path.join(images_dir, args.origin_file)))
	img_tar = rgb2gray(mpimg.imread(os.path.join(images_dir, args.target_file)))
	print("--- Read in input images: %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	# Implement step 1 and step 2 -- calculate the cumulative histograms of both images.
	cdf_org = ecdf(img_org)
	cdf_tar = ecdf(img_tar)
	print("--- Implement step 1 and step 2: %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	# Implement step 3 -- calculate lookup table.
	lktbl = hist_match(cdf_org, cdf_tar)
	print("--- Implement step 3: %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	# Implement step 4 -- map the new intensity of each pixel.
	img_adj = hist_specialize(img_org, lktbl)
	print("--- Implement step 4: %s seconds ---" % (time.time() - start_time))

	# Display results.
	disp((img_org, img_adj, img_tar), args)


if __name__ == '__main__':
	main()
