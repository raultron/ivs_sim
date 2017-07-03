# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:44:51 2017

@author: racuna
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from vision.camera import Camera
from vision.plane import Plane
from ippe import homo2d, ippe
import cv2

#%% Create a camera

cam = Camera()
## Test matrix functions
cam.set_K(794,781,640,480)
cam.set_R(1.0,  0.0,  0.0, np.deg2rad(140.0))

cam_world = np.array([0.0,-2,2,1]).T
cam_t = np.dot(cam.R,-cam_world)

cam.set_t(cam_t[0], cam_t[1],  cam_t[2])
cam.set_P()
cam.img_height = 1280
cam.img_width = 960
P_orig = cam.P
R_orig = cam.R
Rt_orig = cam.Rt



#Create a plane with 4 points to start
pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), grid_size=(2,2), grid_step = 0.5)
pl.update()


#Project points in camera
cam_points = np.array(cam.project(pl.get_points()))

#Add noise in the image
mean = 0 # zero mean
sd = 8 # pixels of standard deviation 
noise = np.random.normal(mean,sd,(2,cam_points.shape[1]))
cam_points[:2,:] = cam_points[:2,:] + noise

# 0 is the mean of the normal distribution you are choosing from
# 1 is the standard deviation of the normal distribution
# 100 is the number of elements you get in array noise


#%% plot
# plot projection
plt.figure()
plt.plot(cam_points[0],cam_points[1],'.',color = pl.get_color(),)
plt.xlim(0,1280)
plt.ylim(0,960)
plt.gca().invert_yaxis()
plt.show()

#%%
#Calculate the pose using ippe
x1 = pl.get_points() # homogeneous 3D coordinates
x2 = cam_points # homogeneous pixel coordinates
x2 = cam.get_normalized_pixel_coordinates(x2) # homogeneous normalized pixel coordinates
out = ippe.mat_run(x1[:3,:],x2[:2,:])

t_error = out['t1'] - cam_t[:3]
print("IPPE - Translation error in each axis", t_error )
print("IPPE - Euclidean distance", sqrt(sum(t_error**2)))


#Calculate the pose using solvepnp
retval, rvec, tvec = cv2.solvePnP(x1[:3,:].T,cam_points[:2,:].T,cam.K, (0))
t_error = tvec.T - cam_t[:3]
print("solvePnP - Translation error in each axis", t_error )
print("solvePnP - Euclidean distance", sqrt(sum(t_error**2)))