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
sys.path.append("..")
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

number_of_points = 4
if number_of_points == 4:
    import gdescent.hpoints_gradient as gd
elif number_of_points == 5:
    import gdescent.hpoints_gradient5 as gd
elif number_of_points == 6:
    import gdescent.hpoints_gradient6 as gd

## Define a Display plane
pl = CircularPlane()
pl.random(n =number_of_points, r = 0.010, min_sep = 0.001)


objectPoints_start = pl.get_points()
############################################################################
#Always the same starting points
#4 points: An ideal square
x1  = 0.15*np.cos(np.deg2rad(45))
y1  = 0.15*np.sin(np.deg2rad(45))
objectPoints_start= np.array(
[[ x1, -x1, -x1,  x1],
 [ y1,  y1, -y1,  -y1],
 [ 0.,       0.,       0.,       0.,     ],
 [ 1.,       1.,       1.,       1.,     ]])

##5 Points: a square and one point in the middle.
#objectPoints_start= np.array(
#[[ x1, -x1, -x1,  x1, 0.],
# [ y1,  y1, -y1,  -y1, 0.],
# [ 0.,       0.,       0.,       0.,  0.   ],
# [ 1.,       1.,       1.,       1.,   1.  ]])
 
##########################################################################
#define the plots
#one Figure for image an object points
fig1 = plt.figure('Image and Object points')
ax_image = fig1.add_subplot(211)
ax_object = fig1.add_subplot(212)

## we create the gradient for the point distribution
normalize= False
n = 0.00000001 #condition number norm 4 points
n = 0.000001 #condition number norm 4 points
#n = 0.0000001 #condition number norm 5 points
gradient = gd.create_gradient(metric='condition_number', n = n)


## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx = 800,fy = 800,cx = 640/2,cy = 480/2)
cam.set_width_heigth(640,480)


#Define a set of angles and a distance
r = 0.8
angles = np.arange(90,91,20)


#START OF THE MAIN LOOP

#List with the results for each camera pose
DataSim = []

for angle in angles:
    #Move the camera to the next pose
    x = r*np.cos(np.deg2rad(angle))
    y = 0
    z = r*np.sin(np.deg2rad(angle))
    cam.set_t(x, y, z)
    cam.set_R_mat(R_matrix_from_euler_t(0.0,0,0))
    cam.look_at([0,0,0])

    #Dictionary with all the important information for one pose
    DataSinglePose = {}
    DataSinglePose['Angle'] = angle
    DataSinglePose['Camera'] = cam.clone()
    DataSinglePose['Iters'] = []
    DataSinglePose['ObjectPoints'] = []
    DataSinglePose['ImagePoints'] = []
    DataSinglePose['CondNumber'] = []

    objectPoints_iter = np.copy(objectPoints_start)
    imagePoints_iter = np.array(cam.project(objectPoints_iter, False))

    gradient = gd.create_gradient(metric='condition_number', n = n)
    for i in range(1000):
        DataSinglePose['ObjectPoints'].append(objectPoints_iter)
        DataSinglePose['ImagePoints'].append(imagePoints_iter)

        input_list = gd.extract_objectpoints_vars(objectPoints_iter)
        input_list.append(np.array(cam.P))
        # TODO Yue: set parameter image_pts_measured as None and append it to input_list
        input_list.append(None)
        mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize = False)

        DataSinglePose['CondNumber'].append(mat_cond)

        #PLOT IMAGE POINTS
        plt.sca(ax_image)
        plt.ion()
        if i==0:
            ax_image.cla()
            ax_image.plot(imagePoints_iter[0],imagePoints_iter[1],'x',color = 'black',)
            ax_image.set_aspect('equal')
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
        plt.pause(0.001)



        ## GRADIENT DESCENT

        objectPoints = np.copy(objectPoints_iter)
        gradient = gd.evaluate_gradient(gradient,objectPoints, np.array(cam.P), normalize)

        objectPoints_iter = gd.update_points(gradient, objectPoints, limitx=0.15,limity=0.15)
        imagePoints_iter = np.array(cam.project(objectPoints_iter, False))
        print "Iteration: ", i
        print "Cond Numb: ", mat_cond
    #objectPoints_start = objectPoints_iter

    print "Angle: ", angle


    DataSim.append(DataSinglePose)


pickle.dump( DataSim, open( "icraSim5points_rotationY.p", "wb" ) )