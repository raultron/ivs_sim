# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:58:52 2017

@author: lracuna
"""

from vision.camera import Camera
from vision.rt_matrix import *
import numpy as np
from matplotlib import pyplot as plt

#%%
# load points
points = np.loadtxt('house.p3d').T
points = np.vstack((points,np.ones(points.shape[1])))


#%%
# setup camera
#P = hstack((eye(3),array([[0],[0],[-10]])))
cam = Camera()
## Test matrix functions
cam.set_K(1460,1460,608,480)
cam.set_width_heigth(1280,960) #TODO Yue
# cam.set_R(0.0,  0.0,  1.0, 0.0)
cam.set_t(0.0,  0.0,  -8.0)
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(140.0)) #TODO Yue
cam.set_P()
print(cam.factor())


#%%


x = np.array(cam.project(points))

#%%
# plot projection
plt.figure()
plt.plot(x[0],x[1],'k.')
plt.xlim(0,1280)
plt.ylim(0,960)
plt.show()

#%%
# create transformation
r = 0.03*np.random.rand(3)
r = np.array([ 0.0,  0.0,  1.0])
t = np.array([ 0.0,  0.0,  0.1])


rot = rotation_matrix(r,0.000)
tras = translation_matrix(t)

#%%
# rotate camera and project
plt.figure()

for i in range(20):
  cam.P = np.dot(cam.P,rot)
  cam.P = np.dot(cam.P,tras)
  x = np.array(cam.project(points))
  plt.plot(x[0],x[1],'.')
  plt.xlim(0,1280)
  plt.ylim(0,960)
  
plt.show()


#Experimental results

#External camera calibration using Physical Chessboard
K_ext = np.array([[492.856172, 0.000000, 338.263513], [0.000000, 526.006429, 257.626108], [0.000000, 0.000000, 1.000000]])
#External camera calibration using Screen Chessboard
K_ext_dyn = np.array([[353.7511506068541, 0, 343.6333596289586], [0, 377.989420116449, 259.6826322930511], [0, 0, 1]])


print (K_ext/K_ext_dyn)