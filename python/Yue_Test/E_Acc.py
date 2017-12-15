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

calc_metrics = False
number_of_points = 4

## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx = 800,fy = 800,cx = 640/2.,cy = 480/2.)
cam.set_width_heigth(640,480)

## The Z axis of camera always looking straight down
cam.rotate_x(np.deg2rad(180.0))
cam.rotate_z(np.deg2rad(-90.0))
#TODO How to set t???
cam.set_t(0.0,0.0,10.0, frame='world')


## Define a Display plane with random initial points
## The center of this plane at (0,0,0) in the world coordinate
pl = CircularPlane()
pl.random(n =number_of_points, r = 0.01, min_sep = 0.01)

#Now we define a distribution of planes on the space
plane_size = (0.3,0.3)

planes = create_cam_distribution(cam, plane_size,
                               theta_params = (0,360,10), phi_params =  (0,70,10),
                               r_params = (0.3,2.0,5), plot=False)
print len(cams)
#%%
new_objectPoints = pl.get_points()