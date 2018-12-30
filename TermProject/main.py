#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : Feature Points Matching
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-12-24
'''

import os, shutil, argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import camera, feature, structure, visualize

# If run in Windows, comment this line.
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--all', action = 'store_const', const =True, 
                    help = 'Specify to run all procedures.')
parser.add_argument('--calibration', action = 'store_const', const = True, 
                    help = 'Whether to calibrate camera. Your chessboard patterns should be stored in ./chessboard_pattern directory.')
parser.add_argument('--undistortion', action = 'store_const', const = True, 
                    help = 'Whether to undistort raw images. Your raw images data should be stored in ./raw_images.')
parser.add_argument('--raw-images', type = str, default = 'IMG', 
                    help = 'The prefix of filenames of raw images. Default \'IMG\'')
parser.add_argument('--resolution', type = int, default = [1280, 960], nargs = '*', 
                    help = 'Image Resolution. The program will resize all images to this specification. Default [1280,960]')
args = parser.parse_args()

if args.all:
    args.calibration = True
    args.undistortion = True
    print('Remove current cache..')
    shutil.rmtree('.tmp')

# Define resolution of input images
assert len(args.resolution) == 2

# Define all paths
chessboard_pattern_path = 'chessboard_pattern'
calibration_result_path = os.path.join('.tmp', 'calibration_result')
camera_para_filename = 'camera_parameters.npz'

raw_images_path = 'raw_images'
undistortion_result_path = os.path.join('.tmp', 'undistorted_images')

keypoints_matching_result_path = os.path.join('.tmp', 'keypoints_matching_result')

view_matrice_path = os.path.join('.tmp', 'view_matrice')

triangulation_path = os.path.join('.tmp', 'triangulation')


def main():

    '''------------------------ Camera Calibration -------------------------------------------------------'''
    print('\n'+'-'*50)
    if args.calibration:
        print('Camera Calibration..')
        camera_matrix, distortion_vector, reprojection_error = camera.camera_calibration(
            chessboard_pattern_path, resolution = args.resolution, number_chessboard_colum = 6, number_chessboard_row = 8, 
            output_images_path = calibration_result_path, output_camera_para_filename = camera_para_filename)
    else: # Load camera parameter
        if not os.path.isfile(os.path.join(calibration_result_path, camera_para_filename)):
            print('Cannot find camera calibration parameters. Please perform camera calibration first.')
            return
        with np.load(os.path.join(calibration_result_path, camera_para_filename)) as reader:
            camera_matrix, distortion_vector, reprojection_error = reader['camera_matrix'], reader['distortion_vector'], reader['reprojection_error']
    
    print('Camera Matrix: \n{}\n'.format(camera_matrix))
    print('Distortion Vector: \n{}\n'.format(distortion_vector))
    print('Reprojection Error: {}'.format(reprojection_error))


    '''------------------------ Undistort Raw Image ------------------------------------------------------'''
    if args.undistortion:
        print('\n'+'-'*50)
        print('Undistort raw images..')
        camera.undistort_images(raw_images_path, camera_matrix, distortion_vector, resolution = args.resolution, output_path = undistortion_result_path)
    else:
        if not (os.path.exists(undistortion_result_path) and os.path.isdir(undistortion_result_path) and os.listdir(undistortion_result_path)):
            print('Cannot find undistorted images. Please perform undistortion first.')
            return

    '''------------------------ Key Points Matching ------------------------------------------------------'''
    print('\n'+'-'*50)
    print('Match key points between two images via SIFT..')
    img1 = cv2.imread(os.path.join(undistortion_result_path, args.raw_images + '_01.jpg'),0)  # queryimage
    img1 = cv2.resize(img1, tuple(args.resolution))
    img2 = cv2.imread(os.path.join(undistortion_result_path, args.raw_images + '_02.jpg'),0)  # trainimage
    img2 = cv2.resize(img2, tuple(args.resolution))
    pts1, pts2 = feature.match_keypoints_between_images(img1, img2, output_path = keypoints_matching_result_path)
    print('{} matches found.'.format(len(pts1)))


    '''--------------------- Fundamental/Essential Matrix Estimation -------------------------------------'''
    print('\n'+'-'*50)
    print('Calculate fundamental/essential matrix..')
    # Perform RANSAC on fundamental matrix estimation
    pts1, pts2, F = structure.fundamentalMat_estimation(pts1, pts2)
    # Calculate essential matrix
    E = np.dot(np.dot(camera_matrix.T, F), camera_matrix)

    print('Fundamental Matrix between two views: \n{}\n'.format(F))
    print('Essential Matrix between two views: \n{}\n'.format(E))
    print('After fundamental matrix estimation, {} inlier matches found.'.format(len(pts1)))

    # Save results
    if not os.path.exists(view_matrice_path):
        os.makedirs(view_matrice_path)
    np.savez(os.path.join(view_matrice_path, 'fundamental_matrix.npz'), F=F)
    np.savez(os.path.join(view_matrice_path, 'essential_matrix.npz'), E=E)

    '''--------------------- Triangulation --------------------------------------------'''
    print('\n'+'-'*50)
    print('Triangulation..')

    # Calculate projection matrix of the first view
    P1 = np.dot(camera_matrix, np.hstack((np.eye(3), np.zeros((3,1)))))
    print('Projection matrix of the first view: \n{}\n'.format(P1))

    # Calculate projection matrix of the second view
    P2s = structure.compute_reprojection_from_essential(E)
    # We will get 4 solutions, so we need to find the correct one
    P2 = structure.find_correct_reprojection(P2s, camera_matrix, P1, pts1.T[:,0], pts2.T[:,0])
    print('Projection matrix of the second view: \n{}\n'.format(P2))

    # Triangulation
    # ATTENTION: all input data should be of float type or it will raise BUS ERROR exception
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts4D /= pts4D[3]
    pts4D = pts4D.T

    # Draw epilines and scatter reconstructed point cloud
    # Generate a color list to instinguish different matches
    colors = [tuple(np.random.randint(0,255,3).tolist()) for i in pts1]
    annotated_img1, annotated_img2 = visualize.drawEpilines(img1, img2, pts1, pts2, F, colors)
    cv2.imwrite(os.path.join(keypoints_matching_result_path, 'IMG01.jpg'), annotated_img1)
    cv2.imwrite(os.path.join(keypoints_matching_result_path, 'IMG02.jpg'), annotated_img2)
    visualize.scatter3DPoints(pts4D, colors, output_path = triangulation_path)


if __name__ == '__main__':
    main()