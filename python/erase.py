#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:24:01 2018

@author: lracuna
"""

import autograd.numpy as np
import autograd.scipy as scipy
from autograd import grad
from autograd import value_and_grad
import gdescent.hpoints_gradient as gd

from vision.camera import *
from vision.circular_plane import CircularPlane
from scipy.optimize import minimize
from autograd.optimizers import adam

def extract_objectpoints_1Darray(objectPoints):
  out = []
  for i in range(objectPoints.shape[1]):
      out.append(objectPoints[0,i])
      out.append(objectPoints[1,i])
  return np.array(out)

## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx = 800,fy = 800,cx = 640,cy = 480)
cam.set_width_heigth(1280,960)

## DEFINE CAMERA POSE LOOKING STRAIGTH DOWN INTO THE PLANE MODEL
#cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(179.0))
#cam.set_t(0.0,-0.0,0.5, frame='world')

cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(179.0))
cam.set_t(0.0,0.0,1.5, frame='world')


## Define a Display plane
pl = CircularPlane(radius=0.1)
pl.random(n =200, r = 0.01, min_sep = 0.001)

objectPoints = pl.get_points()
init_params = extract_objectpoints_1Darray(objectPoints)


def calculate_A_matrix_autograd(params,P,normalize=False):
  P = np.array(cam.P)
  n_points = params.shape[0]/2

  object_pts = np.array([]).reshape([4,0])
  image_pts = np.array([]).reshape([3,0])


  for i in range(n_points):
      x = params[i*2]
      y = params[i*2+1]
      X = np.array([[x],[y],[0.],[1.]]).reshape(4,1)

      U = np.array(np.dot(P,X)).reshape(3,1)

      object_pts = np.hstack([object_pts, X])

      image_pts = np.hstack([image_pts, U])


  if normalize:
    object_pts_norm,T1 = gd.normalise_points(object_pts)
    image_pts_norm,T2 = gd.normalise_points(image_pts)
  else:

    object_pts_norm = object_pts[[0,1,3],:]
    image_pts_norm = image_pts
    image_pts_norm,T2 = gd.normalise_points(image_pts)


  object_pts_norm = object_pts_norm/object_pts_norm[2,:]

  image_pts_norm = image_pts_norm/image_pts_norm[2,:]

  A = np.array([]).reshape([0,9])
  for i in range(n_points):
      x = object_pts_norm[0,i]
      y = object_pts_norm[1,i]

      u = image_pts_norm[0,i]
      v = image_pts_norm[1,i]


      row1 = np.array([ 0,  0, 0, -x, -y, -1,  v*x,  v*y,  v])
      row2 = np.array([x, y, 1,   0,   0,  0, -u*x, -u*y, -u])

      A = np.vstack([A, row1])
      A = np.vstack([A, row2])
  return A


def matrix_condition_number_autograd(params,P,normalize = False):
  A = calculate_A_matrix_autograd(params,P)
  U, s, V = np.linalg.svd(A,full_matrices=False)
  greatest_singular_value = s[0]
#  rcond=1e-5
#  if s[-1] > rcond:
#    smalles_singular_value = s[-1]
#  else:
#    smalles_singular_value = s[-2]
  smallest_singular_value = s[-2]
  return greatest_singular_value/smallest_singular_value
  #return np.sqrt(greatest_singular_value)/np.sqrt(smalles_singular_value)

objective2 = lambda params: matrix_condition_number_autograd(params, cam.P, normalize = False)
objective3 = lambda params,iter: matrix_condition_number_autograd(params, cam.P, normalize = False)

print("Optimizing condition number...")
#out =  minimize(value_and_grad(objective2), init_params, jac=True,
#                      method='COBYLA')
#print out
#optimized_params = out['x']

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
plot = pg.PlotWidget()
s1 = pg.ScatterPlotItem()
plot.show()
def plot_points(params, iter, gradient):
    global s1
    global plot
    x = params[::2]
    y = params[1::2]
    if iter == 0:
        s1 = pg.ScatterPlotItem(x,y)
        plot.addItem(s1)
        plot.show()
    else:
        s1.setData(x*2,y*2)
    QtGui.QApplication.processEvents()



objective_grad = grad(objective3)
optimized_params = adam(objective_grad, init_params, step_size=0.001,
                            num_iters=5000, callback = plot_points)



#plt.scatter(optimized_params[::2], optimized_params[1::2])
#
#
#pg.plot(optimized_params[::2], optimized_params[1::2])
#plt.scatter(init_params[::2], init_params[1::2])
#
#x = optimized_params[::2]
#y = optimized_params[1::2]
#s1 = pg.ScatterPlotItem(x= x,y= y)
