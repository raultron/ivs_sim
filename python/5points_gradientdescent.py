#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:52:11 2017

@author: lracuna
"""
from vision.camera import *
from vision.plane import Plane
import gdescent.hpoints_gradient5 as gd5
from error_functions import geometric_distance_points, get_matrix_conditioning_number, volker_metric,calculate_A_matrix


## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx = 800,fy = 800,cx = 640,cy = 480)
cam.set_width_heigth(1280,960)

## DEFINE CAMERA POSE LOOKING STRAIGTH DOWN INTO THE PLANE MODEL
cam.set_R_axisAngle(1.0,  1.0,  0.0, np.deg2rad(165.0))
cam.set_t(0.0,-0.2,1.0, frame='world')

## Define a Display plane
pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.3,0.3), n = (2,2))
pl.random(n =5, r = 0.01, min_sep = 0.01)

## CREATE A SET OF IMAGE POINTS FOR VALIDATION OF THE HOMOGRAPHY ESTIMATION
validation_plane =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.3,0.3), n = (4,4))
validation_plane.uniform()


## we create the gradient for the point distribution
gradient = gd5.create_gradient(metric='condition_number')



objectPoints_des = pl.get_points()
imagePoints_des = np.array(cam.project(objectPoints_des, False))
objectPoints_list = list()
imagePoints_list = list()
new_objectPoints = objectPoints_des
alpha = 0.001
for i in range(10000):
  objectPoints = np.copy(new_objectPoints)
  gradient = gd5.evaluate_gradient(gradient,objectPoints, np.array(cam.P))
  gradient = gd5.normalize_gradient(gradient)

  new_objectPoints = gd5.update_points(alpha, gradient, objectPoints)
  new_imagePoints = np.array(cam.project(new_objectPoints, False))

  objectPoints_list.append(new_objectPoints)
  imagePoints_list.append(new_imagePoints)
  plt.ion()
  #plt.cla()
  plt.figure('Image Points')
  if i==0:
    plt.cla()
    cam.plot_plane(pl)
    plt.plot(imagePoints_des[0],imagePoints_des[1],'x',color = 'black',)
  plt.plot(new_imagePoints[0],new_imagePoints[1],'.',color = 'blue',)

  plt.xlim(0,1280)
  plt.ylim(0,960)
  plt.gca().invert_yaxis()
  plt.pause(0.01)

  Xo = np.copy(new_objectPoints[[0,1,3],:]) #without the z coordinate (plane)
  Xi = np.copy(new_imagePoints)
  Aideal = calculate_A_matrix(Xo, Xi)

  x1,y1,x2,y2,x3,y3,x4,y4,x5,y5 = gd5.extract_objectpoints_vars(new_objectPoints)
  mat_cond,s = gd5.matrix_conditioning_number_autograd(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,np.array(cam.P))
  print s

  volkerMetric = volker_metric(Aideal)

  print "Iteration: ", i
  print "Mat cond:", mat_cond
  print "Volker Metric:", volkerMetric
  print "dx1,dy1 :", gradient.dx1_eval,gradient.dy1_eval
  print "dx2,dy2 :", gradient.dx2_eval,gradient.dy2_eval
  print "dx3,dy3 :", gradient.dx3_eval,gradient.dy3_eval
  print "dx4,dy4 :", gradient.dx4_eval,gradient.dy4_eval
  print "dx5,dy5 :", gradient.dx5_eval,gradient.dy5_eval
  print "------------------------------------------------------"

