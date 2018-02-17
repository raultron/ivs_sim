# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:15:22 2017

@author: lracuna
"""
from vision.camera import *
from vision.plane import Plane
from vision.screen import Screen

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import cv2


#%% Create a camera

cam = Camera()
## Test matrix functions
cam.set_K(794,781,640,480)
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(175.0))

cam_world = np.transpose(np.array([0.0,0.0,0.6,1]))
cam_t = np.dot(cam.R,-cam_world)

cam.set_t(cam_t[0], cam_t[1],  cam_t[2])
cam.set_P()
cam.img_height = 1280
cam.img_width = 960
P_orig = cam.P
R_orig = cam.R
Rt_orig = cam.Rt



#%%
#Create Screens
s1 = Screen()
s2 = Screen()
s3 = Screen()


#%% Configure Screens
s1.set_origin(np.array([-0.15, 0, 0]))
s1.set_dimensions(0.50,0.30)
s1.set_pixel_pitch(0.04)
red = (1,0,0)
s1.set_color(red)
s1.update()
s1.rotate_z(np.pi/2.0)
s1.rotate_x(np.pi/12.0)
s1.rotate_y(np.pi/4.0)

#s1.rotate_x(pi/2.0)



s2.set_origin(np.array([0.15, 0, 0]))
s2.set_dimensions(0.50,0.30)
s2.set_pixel_pitch(0.04)
green = (0,1,0)
s2.set_color(green)
s2.update()
s2.rotate_z(np.pi/2.0)
s2.rotate_x(np.pi/12.0)
s2.rotate_y(-np.pi/4.0)


s3.set_origin(np.array([0.0, 0.0, 0]))
s3.set_dimensions(0.50,0.30)
s3.set_pixel_pitch(0.04)
yellow = (1,1,0)
s3.set_color(yellow)
s3.update()
#s3.rotate_z(pi/2.0)
s3.rotate_x(np.pi/14.1)
s3.rotate_y(np.pi/10.0)



#%% Project screen points in image
cam_points1 = np.array(cam.project(s1.get_points()))
cam_points2 = np.array(cam.project(s2.get_points()))
cam_points3 = np.array(cam.project(s3.get_points()))


#%% plot

# plot projection
plt.figure()
plt.plot(cam_points1[0],cam_points1[1],'.',color = s1.get_color(),)
plt.plot(cam_points2[0],cam_points2[1],'.',color = s2.get_color(),)
plt.plot(cam_points3[0],cam_points3[1],'.',color = s3.get_color(),)
plt.xlim(0,1280)
plt.ylim(0,960)
plt.gca().invert_yaxis()
plt.show()



#%% Configure 3D plot

# camera in world
cam_world = -np.dot(np.transpose(cam.R[:3,:3]), cam.t[:3])[:,3]

#Camera axis
cam_axis_x = np.transpose(np.array([1,0,0,1]))
cam_axis_y = np.transpose(np.array([0,1,0,1]))
cam_axis_z = np.transpose(np.array([0,0,1,1]))

cam_axis_x = np.dot(np.transpose(cam.R), cam_axis_x)
cam_axis_y = np.dot(np.transpose(cam.R), cam_axis_y)
cam_axis_z = np.dot(np.transpose(cam.R), cam_axis_z)

 
#%% Opencv camera calibration

objp1 = np.transpose(s1.get_points_basis()[:3,:])
imgp1 = np.transpose(cam_points1[:2,:])

objp2 = np.transpose(s2.get_points_basis()[:3,:])
imgp2 = np.transpose(cam_points2[:2,:])

objp3 = np.transpose(s3.get_points_basis()[:3,:])
imgp3 = np.transpose(cam_points3[:2,:])

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

objpoints.append(objp1.astype(np.float32))
objpoints.append(objp2.astype(np.float32))
objpoints.append(objp3.astype(np.float32))
imgpoints.append(imgp2.astype(np.float32))
imgpoints.append(imgp2.astype(np.float32))
imgpoints.append(imgp3.astype(np.float32))

camera_matrix = np.zeros((3, 3))
camera_matrix[0,0]= 750
camera_matrix[1,1]= 750
camera_matrix[0,2]=cam.img_width/2
camera_matrix[1,2]=cam.img_height/2
camera_matrix[2,2]=1.0

distCoeffs= np.zeros(4)


#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (cam.img_width,cam.img_height),None,None)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (cam.img_width,cam.img_height), camera_matrix.astype('float32'),distCoeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

print mtx
print cam_points1[:,0]

#%%
# plot the surface
#plt3d = plt.figure().gca(projection='3d')
#plt3d.set_zlim3d(0, 15)                    # viewrange for z-axis should be [-4,4] 
#plt3d.set_ylim3d(-7, 7)                    # viewrange for y-axis should be [-2,2] 
#plt3d.set_xlim3d(-7, 7)
#
#
#
#
#plt3d.scatter(p1.xx, p1.yy, p1.zz, color = p1.get_color())
#plt3d.quiver(cam_world[0], cam_world[1], cam_world[2], cam_axis_x[0], cam_axis_x[1], cam_axis_x[2], length=0.5, color = 'r', pivot = 'tail')
#plt3d.quiver(cam_world[0], cam_world[1], cam_world[2], cam_axis_y[0], cam_axis_y[1], cam_axis_y[2], length=0.5, color = 'g', pivot = 'tail')
#plt3d.quiver(cam_world[0], cam_world[1], cam_world[2], cam_axis_y[0], cam_axis_y[1], cam_axis_z[2], length=0.5, color = 'b', pivot = 'tail')
#
#plt3d.quiver(0,0,0,1,0,0, length=1.0, color = 'r', pivot = 'tail')
#plt3d.quiver(0,0,0,0,1,0, length=1.0, color = 'g', pivot = 'tail')
#plt3d.quiver(0,0,0,0,0,1, length=1.0, color = 'b', pivot = 'tail')
#plt.show()

#%%
mlab.points3d(s1.plane_points[0], s1.plane_points[1], s1.plane_points[2], scale_factor=0.05, color = s1.get_color())
mlab.points3d(s2.plane_points[0], s2.plane_points[1], s2.plane_points[2], scale_factor=0.05, color = s2.get_color())

mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_x[0], cam_axis_x[1], cam_axis_x[2], line_width=3, scale_factor=0.1, color=(1,0,0))
mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_y[0], cam_axis_y[1], cam_axis_y[2], line_width=3, scale_factor=0.1, color=(0,1,0))
mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_z[0], cam_axis_z[1], cam_axis_z[2], line_width=3, scale_factor=0.1, color=(0,0,1))

mlab.quiver3d(0,0,0,1,0,0, line_width=3, scale_factor=0.1, color=(1,0,0))
mlab.quiver3d(0,0,0,0,1,0, line_width=3, scale_factor=0.1, color=(0,1,0))
mlab.quiver3d(0,0,0,0,0,1, line_width=3, scale_factor=0.1, color=(0,0,1))

mlab.show()



