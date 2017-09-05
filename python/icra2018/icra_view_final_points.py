#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:52:11 2017

@author: Raul Acuna

Description:

The best 4-point configuraiton is obtained by a gradient descent approach
by minimizing the condition number of the A matrix of homography  DLT transform.

We calculate the best point configuration for a set of camera poses.
We position the camera describing a half circle around the plane from 30 to 90 degrees
in intervals of 5 degrees.
For each pose the camera Z axis is aligned to the center of the Plane.
The camera distance to the center of the plane is constant.

"""
import pickle
import sys
sys.path.append("../vision/")
sys.path.append("../gdescent/")
import error_functions as ef
from vision.camera import Camera
from vision.plane import Plane
from vision.circular_plane import CircularPlane
from vision.rt_matrix import R_matrix_from_euler_t
from ippe import homo2d
from homographyHarker.homographyHarker import homographyHarker as hh
from solve_ippe import pose_ippe_both, pose_ippe_best
from solve_pnp import pose_pnp
import cv2
import autograd.numpy as np
import matplotlib.pyplot as plt


DataSim = pickle.load( open( "icraSim5points_rotationY.p", "rb" ) )



## Define a Display plane
pl = CircularPlane()
pl.random(n =number_of_points, r = 0.010, min_sep = 0.001)


#define the plots
#one Figure for image an object points
fig1 = plt.figure('Image and Object points')
ax_image = fig1.add_subplot(211)
ax_object = fig1.add_subplot(212)



#START OF THE MAIN LOOP

for DataSinglePose in DataSim:
    #Dictionary with all the important information for one pose
#    DataSinglePose = {}
#    DataSinglePose['Angle'] = angle
#    DataSinglePose['Camera'] = cam
#    DataSinglePose['Iters'] = []
#    DataSinglePose['ObjectPoints'] = []
#    DataSinglePose['ImagePoints'] = []
#    DataSinglePose['CondNumber'] = []      
    hist = 8
    angle = DataSinglePose['Angle']    
    mat_cond = np.mean(DataSinglePose['CondNumber'][-hist:],0)
    cam = DataSinglePose['Camera']
    ObjectPoints_list = DataSinglePose['ObjectPoints']
    ImagePoints_list = DataSinglePose['ImagePoints'] 
    
    objectPoints_iter = np.mean(ObjectPoints_list[-hist:],0)
    imagePoints_iter = np.mean(ImagePoints_list[-hist:],0)


    #PLOT IMAGE POINTS
    plt.sca(ax_image)
    plt.ion()
    if i==0:
        plt.cla()
        plt.plot(imagePoints_iter[0],imagePoints_iter[1],'x',color = 'black',)
        ax_image.set_aspect('equal', 'datalim')
    ax_image.cla()
    cam.plot_plane(pl)
    ax_image.plot(imagePoints_iter[0],imagePoints_iter[1],'.',color = 'blue',)
      
    ax_image.set_xlim(0,cam.img_width)
    ax_image.set_ylim(0,cam.img_height)
    ax_image.invert_yaxis()
    ax_image.set_title('Image Points')
      
    #PLOT OBJECT POINTS
    plt.sca(ax_object)
    if i==0:
        ax_object.cla()
        ax_object.plot(objectPoints_iter[0],objectPoints_iter[1],'x',color = 'black',)
        ax_object.set_aspect('equal', 'datalim')
    ax_object.cla()
    plt.ion()
    #ax_object.plot(objectPoints_historic[0],objectPoints_historic[1],'.',color = 'blue',)
    ax_object.plot(objectPoints_iter[0],objectPoints_iter[1],'.',color = 'red',)
    pl.plot_plane()
    ax_object.set_title('Object Points')
    ax_object.set_xlim(-pl.radius,pl.radius)
    ax_object.set_ylim(-pl.radius,pl.radius)       

    plt.show()
    plt.pause(0.1)
    
    print "Angle: ", angle
    print "Cond Numb: ", mat_cond