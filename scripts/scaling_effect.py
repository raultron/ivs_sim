# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:59:40 2017

@author: lracuna
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:58:52 2017

@author: lracuna
"""

from camera_calibration import *

#%%
# load points
points = loadtxt('house.p3d').T
points = vstack((points,ones(points.shape[1])))

P1 = array([[0.] ,  [0.], [0.],  [1.]])
P2 = array([[2.] ,  [2.], [0.],  [1.]])
points = hstack((P1,P2))


#%%
# setup real camera
#P = hstack((eye(3),array([[0],[0],[-10]])))
cam = Camera()
fx = 1460
fy = 1460
cx = 608
cy = 480
## Test matrix functions
cam.set_K(fx,fy,cx,cy)
cam.set_R(0.0,  0.0,  1.0, 0.0)
cam.set_t(0.0,  0.0,  -8.0)
cam.set_P()


#Setup wrongly calibrated camera
alpha = 0.8
beta = 1.1
gamma = alpha/beta
bad_cam = Camera()
bad_cam.set_K(alpha*fx,alpha*fy,gamma*cx,gamma*cy)
bad_cam.set_R(0.0,  0.0,  1.0, 0.0)
bad_cam.set_t(0.0,  0.0,  -8.0)
bad_cam.set_P()


#%%
# plot projection real camera
x_good = array(cam.project(points))
plt.figure()
plt.plot(x_good[0],x_good[1],'k.')
plt.xlim(0,1280)
plt.ylim(0,960)
plt.show()

#%%
# plot projection bad camera
x_bad = array(bad_cam.project(points))
plt.figure()
plt.plot(x_bad[0],x_bad[1],'k.')
plt.xlim(0,1280)
plt.ylim(0,960)
plt.show()

#%%
d_good = x_good[:,0] - x_good[:,1]
d_bad = x_bad[:,0] - x_bad[:,1]
d_good/d_bad