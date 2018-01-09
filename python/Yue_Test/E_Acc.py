#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 15.12.17 12:45
@File    : E_Acc.py
@author: Yue Hu
"""
import sys

sys.path.append("../")
from vision.camera import *
from vision.plane import Plane
from vision.circular_plane import CircularPlane
import gdescent.hpoints_gradient as gd
import error_functions as ef
from ippe import homo2d
from homographyHarker.homographyHarker import homographyHarker as hh
from solve_ippe import pose_ippe_both, pose_ippe_best
from solve_pnp import pose_pnp
import cv2
import numpy as np
from plane_distribution import *
from scipy.linalg import expm, rq, det, inv
from DataSimOut import DataSimOut

calc_metrics = False
number_of_points = 4

## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
cam.set_width_heigth(640, 480)

## The Z axis of camera always looking straight down
cam.rotate_x(np.deg2rad(180.0))
cam.rotate_z(np.deg2rad(-90.0))
# TODO How to set t???
cam.set_t(0.0, 0.0, 10.0, frame='world')

## Define a Display plane with random initial points
## The center of this plane at (0,0,0) in the world coordinate
pl = CircularPlane()
pl.random(n=number_of_points, r=0.01, min_sep=0.01)

# Now we define a distribution of planes on the space
plane_size = (0.3, 0.3)

planes = create_plane_distribution(plane_size=(0.3, 0.3), theta_params=(0, 360, 5), phi_params=(0, 70, 3),
                                   r_params=(0.5, 1.0, 2), plot=False)
print "------------Number of Planes-------- "
print len(planes)

# T from static plane at (0,0,0) to static camera straight up
T = cam.R
T[:,3] = cam.t[:,3]

## CREATE A SET OF IMAGE POINTS FOR VALIDATION OF THE HOMOGRAPHY ESTIMATION
# This will create a grid of 16 points of size = (0.3,0.3) meters
validation_plane = Plane(origin=np.array([0, 0, 0]), normal=np.array([0, 0, 1]), size=(0.3, 0.3), n=(4, 4))
validation_plane.uniform()

## we create the gradient for the point distribution
normalize = False
n = 0.000001  # condition number norm 4 points
gradient = gd.create_gradient(metric='condition_number', n=n)

# -------------------------------------------------------------------------------------------
def plane_rotation_angle(cam_xyz=np.array([0,0,10]), pl_origin = np.array([0,0,0]), plane=None, alpha=0, beta=0):
    # TODO need to add R in plane.py, also need to change plot3D_plane method in plane_distribution.py
    plane.rotate_x(alpha)
    plane.rotate_y(beta)
# -------------------------------------------------------------------------------------------


#Results for 4 points ill conditioned
D4pIll = DataSimOut(n,pl,validation_plane, ImageNoise = 4, ValidationIters = 1000)
#Results for 4 points well conditioned (after gradient)
D4pWell = DataSimOut(n,pl,validation_plane, ImageNoise = 4, ValidationIters = 1000)


for pl in planes:
    # TODO for each plane, need to change R t to keep cam not changed!!!!
    #TODO need to change R in plane.py
    T1 = pl.R
    T1[0:3, 3] = pl.origin
    # T = T1 * T2
    # T2 is the translation between new plane and static camera
    T2 = np.dot(inv(T1),T)
    # Create new relationship between new plane and static camera
    new_cam = Camera()
    new_cam.set_K(fx=800, fy=800, cx=640 / 2., cy=480 / 2.)
    new_cam.set_width_heigth(640, 480)
    new_R = np.copy(T2)
    new_R[:,3] = np.array([0,0,0,1])
    new_cam.set_R_mat(new_R)
    new_t = np.copy(T2[0:3,3])
    new_cam.set_t(new_t[0],new_t[1],new_t[2], frame = 'camera')

    ## calculate_metrics
    D4pIll.Camera.append(new_cam)
    D4pIll.ObjectPoints.append(pl.get_points())
    D4pIll.calculate_metrics()

    ## GRADIENT DESCENT
    gradient_iters_max = 100 #TODO  50
    new_objectPoints = pl.get_points()
    for i in range(gradient_iters_max):
        objectPoints = np.copy(new_objectPoints)
        gradient = gd.evaluate_gradient(gradient, objectPoints, np.array(new_cam.P), normalize)
        new_objectPoints = gd.update_points(gradient, objectPoints, limitx=0.15, limity=0.15)
        new_imagePoints = np.array(new_cam.project(new_objectPoints, False))

    D4pWell.Camera.append(new_cam)
    D4pWell.ObjectPoints.append(new_objectPoints)
    D4pWell.calculate_metrics()
    print "---------Plane origin-------"
    print pl.origin
    print "---------CondNumber---------"
    print D4pWell.CondNumber[-1]
# --------------------------------------------------------------------------------------
# print np.median(np.array(D4pIll.Homo_CV_mean)/np.array(D4pWell.Homo_CV_mean))
# print np.median(np.array(D4pIll.pnp_tvec_error_mean)/np.array(D4pWell.pnp_tvec_error_mean))
# print np.median(np.array(D4pIll.CondNumber)/np.array(D4pWell.CondNumber))

print D4pWell.CondNumber