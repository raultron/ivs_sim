# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:44:51 2017

@author: racuna
"""

import numpy as np
import matplotlib.pyplot as plt

from vision.camera import Camera
from vision.plane import Plane
from ippe import homo2d, ippe

#%% Create a camera

cam = Camera()
## Test matrix functions
cam.set_K(794,781,640,480)
cam.set_R(1.0,  0.0,  0.0, np.deg2rad(170.0))

cam_world = np.array([0.0,0.0,5,1]).T
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
x1 = pl.get_points()
x1 = np.delete(x1, 2, axis=0)
x2 = cam_points
H = homo2d.homography2d(x1,x2)


x2 = cam.get_normalized_pixel_coordinates(x2)
out = ippe.mat_run(pl.get_points()[:3,:],x2[:2,:])


