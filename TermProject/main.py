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

# Check display environment
disp = True
if os.name == 'posix' and "DISPLAY" not in os.environ:
    plt.switch_backend('agg')
    disp = False
    
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

pose_estimation_path = os.path.join('.tmp', 'pose_estimation')


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
    print('Match image 1 and image 2..')
    img1 = cv2.imread(os.path.join(undistortion_result_path, args.raw_images + '_01.jpg'),0)
    img1 = cv2.resize(img1, tuple(args.resolution))
    img2 = cv2.imread(os.path.join(undistortion_result_path, args.raw_images + '_02.jpg'),0)
    img2 = cv2.resize(img2, tuple(args.resolution))
    pts12, pts21 = feature.match_keypoints_between_images(img1, img2, output_path = keypoints_matching_result_path)
    print('{} matches found.'.format(len(pts12)))


    '''--------------------- Fundamental/Essential Matrix Estimation -------------------------------------'''
    print('\n'+'-'*50)
    print('Calculate fundamental/essential matrix..')
    # Perform RANSAC on fundamental matrix estimation
    pts12, pts21, F = structure.fundamentalMat_estimation(pts12, pts21)
    # Calculate essential matrix
    E = np.dot(np.dot(camera_matrix.T, F), camera_matrix)

    print('Fundamental Matrix between two views: \n{}\n'.format(F))
    print('Essential Matrix between two views: \n{}\n'.format(E))
    print('After fundamental matrix estimation, {} inlier matches found.'.format(len(pts12)))

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
    P2 = structure.find_correct_reprojection(P2s, camera_matrix, P1, pts12.T[:,0], pts21.T[:,0])
    print('Projection matrix of the second view: \n{}\n'.format(P2))

    # Triangulation
    # ATTENTION: all input data should be of float type or it will raise BUS ERROR exception
    pts3D = cv2.triangulatePoints(P1, P2, pts12.T, pts21.T)
    pts3D /= pts3D[3]
    pts3D = pts3D.T[:,:3]

    # Draw epilines and scatter reconstructed point cloud
    # Generate a color list to instinguish different matches
    colors = [tuple(np.random.randint(0,255,3).tolist()) for i in pts12]
    annotated_img12, annotated_img21 = visualize.drawEpilines(img1, img2, pts12, pts21, F, colors)
    cv2.imwrite(os.path.join(keypoints_matching_result_path, 'IMG01.jpg'), annotated_img12)
    cv2.imwrite(os.path.join(keypoints_matching_result_path, 'IMG02.jpg'), annotated_img21)
    visualize.scatter3DPoints(pts3D, colors, output_path = triangulation_path)
    if disp:
        annotated_img1 = cv2.resize(annotated_img1, (720, 480))
        cv2.imshow('Annotated Image I', annotated_img1)
        annotated_img2 = cv2.resize(annotated_img2, (720, 480))
        cv2.imshow('Annotated Image II', annotated_img2)
        plt.show()

    '''--------------------- Pose Estimation ------------------------------------------'''
    print('\n'+'-'*50)
    print('Pose Estimation..')
    print('Match key points between two images via SIFT..')
    print('Match image 1 and image 3..')
    img3 = cv2.imread(os.path.join(undistortion_result_path, args.raw_images + '_03.jpg'),0)
    img3 = cv2.resize(img3, tuple(args.resolution))
    pts13, pts31 = feature.match_keypoints_between_images(img1, img3, output_path = pose_estimation_path)
    print('{} matches found.'.format(len(pts13)))

    pts13, pts31, F1 = structure.fundamentalMat_estimation(pts13, pts31)
    annotated_img13, annotated_img31 = visualize.drawEpilines(img1, img3, pts13, pts31, F1, colors)
    cv2.imwrite(os.path.join(pose_estimation_path, 'IMG01.jpg'), annotated_img13)
    cv2.imwrite(os.path.join(pose_estimation_path, 'IMG03.jpg'), annotated_img31)

    # Search pts13 that also appear in pts12
    mask = structure.find_common_keypoints(pts12, pts13)
    if (mask.ravel() == -1).all():
        print('Cannot find common matches.')
        return

    # Determine 3D-2D point correspondences on the third view
    map_2Dto3D = {idx2D:idx3D for idx2D, idx3D in enumerate(mask, 1) if idx3D != -1}
    pts3_2D = pts31[list(map_2Dto3D.keys())]
    pts3_3D = pts3D[list(map_2Dto3D.values())]

    print('Common matches: {}'.format(len(pts3_3D)))
    if len(pts3_3D) < 8:
        print('Cannot find enough common matches.')
        return

    # Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
    ret, rvec, t3, __ = cv2.solvePnPRansac(pts3_3D, pts3_2D, camera_matrix, None)
    R3 = cv2.Rodrigues(rvec)[0]
    print('Rotation matrix: \n{}'.format(R3))
    print('Translation: \n{}'.format(t3))

if __name__ == '__main__':
    main()
