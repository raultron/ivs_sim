# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:15:22 2017

@author: lracuna
"""
from camera import *
from plane import Plane
from screen import Screen

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import cv2


#%% Create a camera

cam = Camera()
#Set camera intrinsic parameters
cam.set_K(794,781,640,480)
cam.img_height = 1280
cam.img_width = 960






#%%
#Create 1 Screen
s1 = Screen()
s1.set_origin(np.array([0.0, 0, 0]))
s1.set_dimensions(0.50,0.30)
s1.set_pixel_pitch(0.02)
red = (1,0,0)
s1.set_color(red)
s1.curvature_radius = 1.800
s1.update()
s1.rotate_x(deg2rad(-190.))

#s1.rotate_y(deg2rad(-15.))




#%% Project screen points in image

cam.set_world_position(-0.05, -0.05, -0.45)
cam.rotate_x(deg2rad(+10.))
cam.rotate_y(deg2rad(-5.))
cam.rotate_z(deg2rad(-5.))
cam.set_P()
cam_points1 = array(cam.project(s1.get_points()))



#%% plot

# plot projection
plt.figure()
plt.plot(cam_points1[0],cam_points1[1],'.',color = 'r')
plt.xlim(0,1280)
plt.ylim(0,960)
plt.gca().invert_yaxis()
plt.show()




#%% Opencv camera calibration

objp1 = transpose(s1.get_points_basis()[:3,:])
imgp1 = transpose(cam_points1[:2,:])



#%% Build A matrix for DLT Am = 0
# m = transpose([m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34])
# A = [[Xi, Yi, Zi, 1, 0, 0, 0, 0, -ui*Xi, -ui*Yi, -ui*Zi, -ui],
#      [0, 0, 0, 0, Xi, Yi, Zi, 1, -vi*Xi, -vi*Yi, -vi*Zi, -vi]]
# [Xi,Yi,Zi] point i in world coordinates (meters)
# [ui, vi] projection of point i in image plane (pixels) 
n = len(objp1)
A = empty(shape=[2*n,12])
b = zeros(shape=[2*n,1])
for i in range(n):
    Xi = objp1[i][0]
    Yi = objp1[i][1]
    Zi = objp1[i][1]
    ui = imgp1[i][0]
    vi = imgp1[i][1]
    
    A[(i*2)] =   array([Xi, Yi, Zi, 1, 0, 0, 0, 0, -ui*Xi, -ui*Yi, -ui*Zi, -ui])
    A[(i*2)+1] = array([0, 0, 0, 0, Xi, Yi, Zi, 1, -vi*Xi, -vi*Yi, -vi*Zi, -vi])


U, s, Vh = np.linalg.svd(dot(transpose(A),A), full_matrices=False)
V = Vh.T
#m, _, _, _ = np.linalg.lstsq(dot(transpose(A),A), b)
#%%


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

objpoints.append(objp1.astype(np.float32))
#objpoints.append(objp2.astype(np.float32))
#objpoints.append(objp3.astype(np.float32))

#With round we introduce quatization error
sim_quant_error = True
if sim_quant_error == True:
    imgpoints.append(imgp1.round().astype(np.float32))
    #imgpoints.append(imgp2.round().astype(np.float32))
    #imgpoints.append(imgp3.round().astype(np.float32))
else:
    imgpoints.append(imgp1.astype(np.float32))
    #imgpoints.append(imgp2.astype(np.float32))
    #imgpoints.append(imgp3.astype(np.float32))
    

camera_matrix = np.zeros((3, 3))
camera_matrix[0,0]= 500
camera_matrix[1,1]= 500
camera_matrix[0,2]=cam.img_width/2
camera_matrix[1,2]=cam.img_height/2
camera_matrix[2,2]=1.0

distCoeffs= np.zeros(4)


#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (cam.img_width,cam.img_height),None,None)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (cam.img_width,cam.img_height), camera_matrix.astype('float32'),distCoeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

print mtx
print cam_points1[:,0]