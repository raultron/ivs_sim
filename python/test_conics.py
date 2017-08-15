#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:36:30 2017

@author: lracuna
"""
from vision.conics import Circle, Ellipse
from pose_sim import *
from vision.camera import *
from vision.plane import Plane
from vision.screen import Screen
from ippe import homo2d


import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from Rt_matrix_from_euler_t import R_matrix_from_euler_t
from uniform_sphere import uniform_sphere

#############################################
## INITIALIZATIONS
#############################################

## CREATE A SIMULATED CAMERA
cam = Camera()
fx = fy =  800
cx = 640
cy = 480
cam.set_K(fx,fy,cx,cy)
cam.img_width = 1280
cam.img_height = 960

## DEFINE CAMERA POSE LOOKING STRAIGTH DOWN INTO THE PLANE MODEL
cam.set_R_axisAngle(1.0, 0.0,  0.0, np.deg2rad(130.0))
cam_world = np.array([0.0,0.0,1,1]).T
cam_t = np.dot(cam.R,-cam_world)
cam.set_t(cam_t[0], cam_t[1],  cam_t[2])
cam.set_P()
H_cam = cam.homography_from_Rt()

c1 = Circle((0,0),r=0.05)
print c1.calculate_center()
#c1.contour()

c2 = Ellipse((0,0))
print c2.calculate_center()
#c2.contour()


c3 = c1.project(H_cam)
c3.contour(grid_size=2000)
print c3.calculate_center()

point = np.array([0,0,0,1]).reshape(4,1)
center_circle = cam.project(point)
print center_circle