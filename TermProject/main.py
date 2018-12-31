#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : main
Author          : Charles Young
Python Version  : Python 3.6.3
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-12-30
'''

import os, shutil, argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import camera, feature, structure, visualize, optimal

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
parser.add_argument('--raw-images', type = str, default = 'desk', 
                    help = 'The prefix of filenames of raw images. Default \'desk\'')
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

bundle_adjustment_path = os.path.join('.tmp', 'bundle_adjustment')


def main():

    '''------------------------ Camera Calibration -------------------------------------------------------'''
    print('\n'+'-'*50)
    if args.calibration:
        print('Camera Calibration..')
        intrinsic_matrix, distortion_vector, reprojection_error = camera.camera_calibration(
            chessboard_pattern_path, resolution = args.resolution, number_chessboard_colum = 6, number_chessboard_row = 8, 
            output_images_path = calibration_result_path, output_camera_para_filename = camera_para_filename)
    else: # Load camera parameter
        if not os.path.isfile(os.path.join(calibration_result_path, camera_para_filename)):
            print('Cannot find camera calibration parameters. Please perform camera calibration first.')
            return
        with np.load(os.path.join(calibration_result_path, camera_para_filename)) as reader:
            intrinsic_matrix, distortion_vector, reprojection_error = reader['intrinsic_matrix'], reader['distortion_vector'], reader['reprojection_error']
    
    print('Camera Matrix: \n{}\n'.format(intrinsic_matrix))
    print('Distortion Vector: \n{}\n'.format(distortion_vector))
    print('Reprojection Error: {}'.format(reprojection_error))


    '''------------------------ Undistort Raw Image ------------------------------------------------------'''
    if args.undistortion:
        print('\n'+'-'*50)
        print('Undistort raw images..')
        camera.undistort_images(raw_images_path, intrinsic_matrix, distortion_vector, resolution = args.resolution, output_path = undistortion_result_path)
    
    img1 = cv2.imread(os.path.join(undistortion_result_path, args.raw_images + '_01.jpg'),0)
    img2 = cv2.imread(os.path.join(undistortion_result_path, args.raw_images + '_02.jpg'),0)
    img3 = cv2.imread(os.path.join(undistortion_result_path, args.raw_images + '_03.jpg'),0)
    if img1 is None or img2 is None or img3 is None:
        print('Cannot find undistorted images. Please perform undistortion first.')
        return 
    img1 = cv2.resize(img1, tuple(args.resolution))
    img2 = cv2.resize(img2, tuple(args.resolution))
    img3 = cv2.resize(img3, tuple(args.resolution))


    '''------------------------ Key Points Matching ------------------------------------------------------'''
    print('\n'+'-'*50)
    print('Match key points between two images via SIFT..')

    print('Match image 1 and image 2..')
    pts12, pts21 = feature.match_keypoints_between_images(img1, img2, 
        output_path = keypoints_matching_result_path, inlier_points_filename = 'inliers_12.npz')
    print('{} matches found.'.format(len(pts12)))

    print('Match image 1 and image 3..')
    pts13, pts31 = feature.match_keypoints_between_images(img1, img3, 
        output_path = keypoints_matching_result_path, inlier_points_filename = 'inliers_13.npz')
    print('{} matches found.'.format(len(pts13)))

    print('Match image 2 and image 3..')
    pts23, pts32 = feature.match_keypoints_between_images(img2, img3, 
        output_path = keypoints_matching_result_path, inlier_points_filename = 'inliers_23.npz')
    print('{} matches found.'.format(len(pts23)))


    '''--------------------- Fundamental/Essential Matrix Estimation -------------------------------------'''
    print('\n'+'-'*50)
    print('Calculate fundamental/essential matrix between image 1 and image 2..')
    # Perform RANSAC on fundamental matrix estimation
    pts12, pts21, F = structure.fundamentalMat_estimation(pts12, pts21)
    # Calculate essential matrix
    E = np.dot(np.dot(intrinsic_matrix.T, F), intrinsic_matrix)

    print('Fundamental Matrix between two views (1 and 2): \n{}\n'.format(F))
    print('Essential Matrix between two views (1 and 2): \n{}\n'.format(E))
    print('After fundamental matrix estimation, {} inlier matches remain.'.format(len(pts12)))


    '''--------------------- Triangulation --------------------------------------------'''
    print('\n'+'-'*50)
    print('Triangulation..')

    # Calculate projection matrix of the first view
    P1 = np.dot(intrinsic_matrix, np.hstack((np.eye(3), np.zeros((3,1)))))
    print('Projection matrix of the first view: \n{}\n'.format(P1))

    # Calculate projection matrix of the second view
    # First calculate extrinsic matrix ([R|t]) from essential matrix
    P2s = structure.compute_extrinsic_from_essential(E)
    # We will get 4 solutions, so we need to find the correct one
    P2 = structure.find_correct_projection(P2s, intrinsic_matrix, P1, pts12.T[:,0], pts21.T[:,0])
    print('Projection matrix of the second view: \n{}\n'.format(P2))

    # Triangulation
    # ATTENTION: all input data should be of float type or it will raise BUS ERROR exception
    pts3D = cv2.triangulatePoints(P1, P2, pts12.T, pts21.T)
    pts3D /= pts3D[3]
    pts3D = pts3D.T[:,:3]

    if not os.path.exists(triangulation_path):
        os.makedirs(triangulation_path)
    np.savez(os.path.join(triangulation_path, 'reconstrcuted_3d_points.npz'), pts=pts3D)    


    '''--------------------- Pose Estimation ------------------------------------------'''
    print('\n'+'-'*50)
    print('Pose Estimation..')

    print('Match key points between two images via SIFT..')
    print('Match image 1 and image 3..')
    pts3_2D_1, pts3_3D_1 = structure.find_3Dto2D_point_correspondences(pts13, pts31, pts3D, pts12)
    print('Find {} common matches.'.format(len(pts3_2D_1)))

    print('Match image 2 and image 3..')
    pts3_2D_2, pts3_3D_2 = structure.find_3Dto2D_point_correspondences(pts23, pts32, pts3D, pts21)
    print('Find {} common matches.'.format(len(pts3_2D_2)))

    # Determine 3D-2D point correspondences on the third view
    pts3_2D = pts3_2D_1
    pts3_3D = pts3_3D_1
    for i, row in enumerate(pts3_2D_2, 1):
        idx = np.where((pts3_2D == row).all(1))[0]
        if idx.size:
            continue
        pts3_2D = np.vstack([pts3_2D, row])
        pts3_3D = np.vstack([pts3_3D, pts3_3D_2[i-1]])

    print('Common matches: {}'.format(len(pts3_3D)))
    if len(pts3_3D) < 8:
        print('Cannot find enough common matches.')
        return

    # Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
    __, rvec, t3, __ = cv2.solvePnPRansac(pts3_3D, pts3_2D, intrinsic_matrix, None)
    R3 = cv2.Rodrigues(rvec)[0]
    P3 = np.dot(intrinsic_matrix, np.hstack((R3, t3)))
    print('Projection matrix of the third view: \n{}'.format(P3))

    # Update 3D-2D point correspondences on the first and second views
    pts1_2D = []
    pts2_2D = []
    for pt in pts3_3D:
        idx = np.where((pts3D == pt).all(1))[0]
        pts1_2D.append(pts12[idx][0])
        pts2_2D.append(pts21[idx][0])
    pts1_2D = np.asarray(pts1_2D)
    pts2_2D = np.asarray(pts2_2D)
    pts3D = pts3_3D

    if not os.path.exists(pose_estimation_path):
        os.makedirs(pose_estimation_path)
    np.savez(os.path.join(pose_estimation_path, 'camera_projection_matrix.npz'), P1=P1, P2=P2, P3=P3)    
    np.savez(os.path.join(pose_estimation_path, '3D_2D_points_correspondences.npz'), D3=pts3D, D2_1=pts1_2D, D2_2=pts2_2D, D2_3=pts3_2D) 

    if 0:
        with np.load(os.path.join(pose_estimation_path, '3D_2D_points_correspondences.npz')) as reader:
            pts3D, pts1_2D, pts2_2D, pts3_2D = reader['D3'], reader['D2_1'], reader['D2_2'], reader['D2_3']
        with np.load(os.path.join(pose_estimation_path, 'camera_projection_matrix.npz')) as reader:
            P1, P2, P3 = reader['P1'], reader['P2'], reader['P3']


    '''--------------------- Bundle Adjustment ------------------------------------------'''
    print('\n'+'-'*50)
    print('Bundle Adjustment..')    
    print('Reprojection error on the first view : {}'.format(optimal.reprojection_error(pts3D, pts1_2D, P1, None)))
    print('Reprojection error on the second view : {}'.format(optimal.reprojection_error(pts3D, pts2_2D, P2, None)))
    print('Reprojection error on the third view : {}'.format(optimal.reprojection_error(pts3D, pts3_2D, P3, None)))

    # Prepare input parameters for sba
    n_cameras = 3
    n_points = len(pts3D)
    camera_poses = optimal.extract_pose_from_projection(np.asarray([[P1], [P2], [P3]]))
    pts2D = np.vstack([pts1_2D, pts2_2D, pts3_2D])
    camera_indices = np.arange(n_cameras).repeat(n_points)
    point_indices = np.tile(np.arange(n_points), n_cameras)

    # Optimization with sba
    sba = optimal.bundle_adjustment(camera_poses, pts3D, pts2D, camera_indices, point_indices, intrinsic_matrix)
    optimal_poses, optimal_pts3D = sba.optimize()

    # Extract projection matrice from the results
    optimal_P1, optimal_P2, optimal_P3 = optimal.calculate_projection_from_pose(optimal_poses, intrinsic_matrix)

    if not os.path.exists(bundle_adjustment_path):
        os.makedirs(bundle_adjustment_path)
    np.savez(os.path.join(bundle_adjustment_path, 'camera_projection_matrix.npz'), P1=optimal_P1, P2=optimal_P2, P3=optimal_P3)    
    np.savez(os.path.join(bundle_adjustment_path, '3D_2D_points_correspondences.npz'), D3=optimal_pts3D, D2_1=pts1_2D, D2_2=pts2_2D, D2_3=pts3_2D) 

    print('Reprojection error on the first view : {}'.format(optimal.reprojection_error(optimal_pts3D, pts1_2D, optimal_P1, None)))
    print('Reprojection error on the second view : {}'.format(optimal.reprojection_error(optimal_pts3D, pts2_2D, optimal_P2, None)))
    print('Reprojection error on the third view : {}'.format(optimal.reprojection_error(optimal_pts3D, pts3_2D, optimal_P3, None)))

    if 0:
        with np.load(os.path.join(bundle_adjustment_path, '3D_2D_points_correspondences.npz')) as reader:
            optimal_pts3D, pts1_2D, pts2_2D, pts3_2D = reader['D3'], reader['D2_1'], reader['D2_2'], reader['D2_3']
        with np.load(os.path.join(bundle_adjustment_path, 'camera_projection_matrix.npz')) as reader:
            optimal_P1, optimal_P2, optimal_P3 = reader['P1'], reader['P2'], reader['P3']


    '''--------------------- Visualization ------------------------------------------'''
    # Generate a color list to instinguish different matches
    colors = [tuple(np.random.randint(0,255,3).tolist()) for i in range(len(optimal_pts3D))]
    # Draw epilines and scatter reconstructed point cloud
    __, __, F12 = structure.fundamentalMat_estimation(pts1_2D, pts2_2D)
    annotated_img12, annotated_img21 = visualize.drawEpilines(img1, img2, pts1_2D, pts2_2D, F12, colors)
    __, __, F13 = structure.fundamentalMat_estimation(pts1_2D, pts3_2D)
    annotated_img13, annotated_img31 = visualize.drawEpilines(img1, img3, pts1_2D, pts3_2D, F13, colors)
    __, __, F23 = structure.fundamentalMat_estimation(pts2_2D, pts3_2D)
    annotated_img23, annotated_img32 = visualize.drawEpilines(img2, img3, pts2_2D, pts3_2D, F23, colors)
    if not os.path.exists(keypoints_matching_result_path):
        os.makedirs(keypoints_matching_result_path)
    cv2.imwrite(os.path.join(keypoints_matching_result_path, 'IMG01_2.jpg'), annotated_img12)
    cv2.imwrite(os.path.join(keypoints_matching_result_path, 'IMG02_1.jpg'), annotated_img21)
    cv2.imwrite(os.path.join(keypoints_matching_result_path, 'IMG01_3.jpg'), annotated_img13)
    cv2.imwrite(os.path.join(keypoints_matching_result_path, 'IMG03_1.jpg'), annotated_img31)
    cv2.imwrite(os.path.join(keypoints_matching_result_path, 'IMG02_3.jpg'), annotated_img23)
    cv2.imwrite(os.path.join(keypoints_matching_result_path, 'IMG03_2.jpg'), annotated_img32)
    visualize.scatter3DPoints(optimal_pts3D, colors, output_path = triangulation_path)
    if disp:
        annotated_img12 = cv2.resize(annotated_img12, (720, 480))
        cv2.imshow('Annotated Image I - II', annotated_img12)
        annotated_img21 = cv2.resize(annotated_img21, (720, 480))
        cv2.imshow('Annotated Image II - I', annotated_img21)
        annotated_img12 = cv2.resize(annotated_img13, (720, 480))
        cv2.imshow('Annotated Image I - III', annotated_img13)
        annotated_img21 = cv2.resize(annotated_img31, (720, 480))
        cv2.imshow('Annotated Image III - I', annotated_img31)
        annotated_img12 = cv2.resize(annotated_img23, (720, 480))
        cv2.imshow('Annotated Image II - III', annotated_img23)
        annotated_img21 = cv2.resize(annotated_img32, (720, 480))
        cv2.imshow('Annotated Image III - II', annotated_img32)
        plt.show()


if __name__ == '__main__':
    main()
