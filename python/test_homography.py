# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:30:20 2017

@author: racuna
"""

from vision.camera import *
from vision.plane import Plane
from vision.screen import Screen
from ippe import homo2d

import numpy as np
import matplotlib.pyplot as plt


#%% Create a camera

cam = Camera()
## Test matrix functions
cam.set_K(794,781,640,480)
cam.set_R(1.0,  0.0,  0.0, deg2rad(170.0))

cam_world = transpose(array([0.0,0.0,5,1]))
cam_t = dot(cam.R,-cam_world)

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
cam_points = array(cam.project(pl.get_points()))


#%% plot

# plot projection
plt.figure()
plt.plot(cam_points[0],cam_points[1],'.',color = pl.get_color(),)
plt.xlim(0,1280)
plt.ylim(0,960)
plt.gca().invert_yaxis()
plt.show()

#%%
#Calculate the homography
x1 = pl.get_points()
x1 = np.delete(x1, 2, axis=0)
x2 = cam_points
H = homo2d.homography2d(x1,x2)

#Confirm homography
x_test = dot(H,x1)

#Normalize points
for i in range(shape(x_test)[1]):
    x_test[:,i] = x_test[:,i]/x_test[2,i]

print error = x_test - x2