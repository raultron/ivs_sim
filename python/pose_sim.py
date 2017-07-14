#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:23:20 2017

@author: lracuna
"""

import numpy as np
import cv2

import matplotlib.pyplot as plt
from ippe import homo2d

from vision.rt_matrix import rot_matrix_error
from vision.camera import Camera
from vision.plane import Plane

from mayavi import mlab
from solve_pnp import pose_pnp
from solve_ippe import pose_ippe_both, pose_ippe_best

def show_homo2d_normalization(imagePoints):
    imagePoints_normalized = homo2d.normalise2dpts(imagePoints)
    imagePoints_normalized = imagePoints_normalized[0]
    plt.figure()
    plt.plot(imagePoints_normalized[0],imagePoints_normalized[1],'.',color = 'red',)
    plt.plot(imagePoints_normalized[0,0],imagePoints_normalized[1,0],'.',color = 'blue')
    plt.gca().invert_yaxis()
    plt.show()

def calc_estimated_pose_error(tvec_ref, rmat_ref, tvec_est, rmat_est):
    # Translation error percentual
    tvec_error = np.linalg.norm(tvec_est[:3] - tvec_ref[:3])/np.linalg.norm(tvec_ref[:3])*100.

    #tvec_error = np.sqrt((np.sum((tvec_est[:3]- tvec_ref[:3])**2))

    #Rotation matrix error
    rmat_error = rot_matrix_error(rmat_ref,rmat_est, method = 'angle')
    #rmat_error = rot_matrix_error(rmat_ref,rmat_est)
    return tvec_error, rmat_error

def plot3D_cam(cam, axis_scale = 0.2):
    #Coordinate Frame of real camera
    #Camera axis
    cam_axis_x = np.array([1,0,0,1]).T
    cam_axis_y = np.array([0,1,0,1]).T
    cam_axis_z = np.array([0,0,1,1]).T

    cam_axis_x = np.dot(cam.R.T, cam_axis_x)
    cam_axis_y = np.dot(cam.R.T, cam_axis_y)
    cam_axis_z = np.dot(cam.R.T, cam_axis_z)

    cam_world = cam.get_world_position()

    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_x[0], cam_axis_x[1], cam_axis_x[2], line_width=3, scale_factor=axis_scale, color=(1-axis_scale,0,0))
    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_y[0], cam_axis_y[1], cam_axis_y[2], line_width=3, scale_factor=axis_scale, color=(0,1-axis_scale,0))
    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_z[0], cam_axis_z[1], cam_axis_z[2], line_width=3, scale_factor=axis_scale, color=(0,0,1-axis_scale))


def plot3D(cams, planes):
    #mlab.figure(figure=None, bgcolor=(0.1,0.5,0.5), fgcolor=None, engine=None, size=(400, 350))
    axis_scale = 0.05
    for cam in cams:
        plot3D_cam(cam, axis_scale)
        #axis_scale = axis_scale - 0.1

    for plane in planes:
        #Plot plane points in 3D
        plane_points = plane.get_points()
        mlab.points3d(plane_points[0], plane_points[1], plane_points[2], scale_factor=0.05, color = plane.get_color())
        mlab.points3d(plane_points[0,0], plane_points[1,0], plane_points[2,0], scale_factor=0.05, color = (0.,0.,1.))

    mlab.show()

def run_single(cam, objectPoints, noise = 0, quant_error = False, plot = False, debug = False):
    #Project points in camera

    imagePoints = np.array(cam.project(objectPoints,quant_error))
    #Add Gaussian noise in pixel coordinates
    if noise:
      imagePoints = cam.addnoise_imagePoints(imagePoints, mean = 0, sd = noise)

    #Calculate the pose using solvepnp
    pnp_tvec, pnp_rmat = pose_pnp(objectPoints, imagePoints, cam.K, debug, cv2.SOLVEPNP_ITERATIVE,False)
    pnpCam = cam.clone_withPose(pnp_tvec, pnp_rmat)

    #Calculate the pose using IPPE (solution with least repro error)
    normalizedimagePoints = cam.get_normalized_pixel_coordinates(imagePoints)
    ippe_tvec, ippe_rmat = pose_ippe_best(objectPoints, normalizedimagePoints, debug)
    ippeCam = cam.clone_withPose(ippe_tvec, ippe_rmat)

    #Calculate the pose using IPPE (both solutions)
    #ippe_tvec1,ippe_rmat1,ippe_tvec2,ippe_rmat2 = pose_ippe_both(objectPoints, normalizedimagePoints, debug)
    #ippeCam1 = cam.clone_withPose(ippe_tvec1, ippe_rmat1)
    #ippeCam2 = cam.clone_withPose(ippe_tvec2, ippe_rmat2)

    #Calculate errors
    pnp_tvec_error, pnp_rmat_error = calc_estimated_pose_error(cam.get_tvec(), cam.R, pnpCam.get_tvec(), pnp_rmat)
    ippe_tvec_error, ippe_rmat_error = calc_estimated_pose_error(cam.get_tvec(), cam.R, ippeCam.get_tvec(), ippe_rmat)

    if debug:
      # Print errors

      # IPPE pose estimation errors
      print ("----------------------------------------------------")
      print ("Translation Errors")
      print("IPPE    : %f" % ippe_tvec_error)
      print("solvePnP: %f" % pnp_tvec_error)

      # solvePnP pose estimation errors
      print ("----------------------------------------------------")
      print ("Rotation Errors")
      print("IPPE    : %f" % ippe_rmat_error)
      print("solvePnP: %f" % pnp_rmat_error)

    if plot:
      #Projected image points with ground truth camera
      cam.plot_image(imagePoints, 'r')

      #Projected image points with pnp pose estimation
      #pnpCam.plot_image(imagePoints, 'r')

      #Projected image points with ippe pose estimation
      #Show the effect of the homography normalization
      #show_homo2d_normalization(imagePoints)
      #ippeCam.plot_image(imagePoints, 'r')

      #Show camera frames and plane poses in 3D
      #cams = [cam, ippeCam1, ippeCam2, pnpCam]
      #planes = [pl]
      #plot3D(cams, planes)

    return ippe_tvec_error, ippe_rmat_error, pnp_tvec_error, pnp_rmat_error








def run_point_distribution_test(cam, objectPoints, plot=True):
    #Project points in camera
    imagePoints = np.array(cam.project(objectPoints,quant_error=False))

    #Add Gaussian noise in pixel coordinates
    #imagePoints = addnoise_imagePoints(imagePoints, mean = 0, sd = 2)

    #Show projected points
    if plot:
      cam.plot_image(imagePoints, pl.get_color())

    #Show the effect of the homography normalization
    if plot:
      show_homo2d_normalization(imagePoints)

    #Calculate the pose using solvepnp and plot the image points
    pnp_tvec, pnp_rmat = pose_pnp(objectPoints, imagePoints, cam.K)
    pnpCam = cam.clone_withPose(pnp_tvec, pnp_rmat)
    #if plot:
    #  pnpCam.plot_image(imagePoints, pl.get_color())

    #Calculate the pose using IPPE and plot the image points
    normalizedimagePoints = cam.get_normalized_pixel_coordinates(imagePoints)
    ippe_tvec, ippe_rmat = pose_ippe_best(objectPoints, normalizedimagePoints)

    ippeCam = cam.clone_withPose(ippe_tvec, ippe_rmat)
    #if plot:
    #  ippeCam.plot_image(imagePoints, pl.get_color())


    ippe_tvec1,ippe_rmat1,ippe_tvec2,ippe_rmat2 = pose_ippe_both(objectPoints, normalizedimagePoints)
    ippeCam1 = cam.clone_withPose(ippe_tvec1, ippe_rmat1)
    ippeCam2 = cam.clone_withPose(ippe_tvec2, ippe_rmat2)

    print "--------------------------------------------------"
    print ippe_tvec1
    print ippe_tvec2
    print pnp_tvec
    print cam.get_tvec()
    print "--------------------------------------------------"



    #Calculate errors
    pnp_tvec_error, pnp_rmat_error = calc_estimated_pose_error(cam.get_tvec(), cam.R, pnpCam.get_tvec(), pnp_rmat)
    ippe_tvec_error, ippe_rmat_error = calc_estimated_pose_error(cam.get_tvec(), cam.R, ippeCam.get_tvec(), ippe_rmat)

    # Print errors

    # IPPE pose estimation errors
    print ("----------------------------------------------------")
    print ("Translation Errors")
    print("IPPE    : %f" % ippe_tvec_error)
    print("solvePnP: %f" % pnp_tvec_error)

    # solvePnP pose estimation errors
    print ("----------------------------------------------------")
    print ("Rotation Errors")
    print("IPPE    : %f" % ippe_rmat_error)
    print("solvePnP: %f" % pnp_rmat_error)

    if plot:
      cams = [cam,ippeCam, pnpCam]
      planes = [pl]
      plot3D(cams, planes)


    return cam, pl