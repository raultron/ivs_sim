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
#cam.set_K(fx = 1000,fy = 1000,cx = 640,cy = 480)
#cam.set_width_heigth(1280,960)
w = 1280
h = 960
fx = 1800
fy = 1800

cam.set_K(fx = fx,fy = fy,cx = w/2.,cy = h/2.)
cam.set_width_heigth(w,h)

## DEFINE CAMERA POSE LOOKING STRAIGTH DOWN INTO THE PLANE MODEL
# Initial camera pose looking stratight down into the center of in different orientations plane model """

cam.set_t(0, 0,2)
cam.set_R_mat(R_matrix_from_euler_t(0.0,0,0))


cam.look_at([0,0,0])



cam.set_P()

P = cam.P
R= np.mat(cam.R[:3,:3])
t = np.mat(cam.t[:3].reshape(3,1))
Rt = cam.Rt

H_cam = cam.homography_from_Rt()

c1 = Circle((0,0.1),r=0.05)
c1.contour()
print (c1.calculate_center())

c2 = Circle((2*0.05+ 0.05,0),r=0.05)

#c2 = Ellipse((0,0))
#c2.contour(grid_size=100)
#print (c2.calculate_center())


c3 = c1.project(H_cam)
c3.contour(grid_size=100)
print (c3.calculate_center())
#print c3.major_axis_length()

point = np.array([c1.calculate_center()[0],c1.calculate_center()[1],0.,1.]).reshape(4,1)
center_circle = cam.project(point)
print (center_circle)

cam.plot_image(center_circle)

print cam.get_world_position()

cam.plot_image(np.array(c3.calculate_center()), points_color='red')

### Project circle points
v = c1.contour_data()
for i in np.arange(v.shape[0]):
    point_circle = np.array([v[i][0],v[i][1],0.,1.]).reshape(4,1)
    point_circle_projected = cam.project(point_circle)
    cam.plot_image(point_circle_projected)
    

### Project circle points
v = c2.contour_data()
for i in np.arange(v.shape[0]):
    point_circle = np.array([v[i][0],v[i][1],0.,1.]).reshape(4,1)
    point_circle_projected = cam.project(point_circle)
    cam.plot_image(point_circle_projected)


print("Error")
print(np.sqrt((center_circle[0,0]-c3.calculate_center()[0])**2 + (center_circle[1,0]-c3.calculate_center()[1])**2))

point = np.array([0,0,0.,1.]).reshape(4,1)
x1 = cam.project(point)

X = np.mat([0,0,0]).T
x = cam.K*(R*X + t)
x = x/x[2]

print x1
print x