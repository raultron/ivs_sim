#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:52:11 2017

@author: lracuna
"""
from vision.camera import *
from vision.plane import Plane
import gdescent.hpoints_gradient5 as gd5
import error_functions as ef
from ippe import homo2d


## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx = 800,fy = 800,cx = 640,cy = 480)
cam.set_width_heigth(1280,960)

## DEFINE CAMERA POSE LOOKING STRAIGTH DOWN INTO THE PLANE MODEL
#cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
#cam.set_t(0.0,-0.0,0.5, frame='world')

cam.set_R_axisAngle(1.0,  1.0,  0.0, np.deg2rad(130.0))
cam.set_t(0.0,-0.4,2.0, frame='world')

## Define a Display plane
pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.3,0.3), n = (2,2))
pl.random(n =5, r = 0.001, min_sep = 0.001)

## CREATE A SET OF IMAGE POINTS FOR VALIDATION OF THE HOMOGRAPHY ESTIMATION
validation_plane =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.5,0.5), n = (4,4))
validation_plane.uniform()




## we create the gradient for the point distribution

gradient = gd5.create_gradient(metric='condition_number')
#gradient = gd5.create_gradient(metric='volker_metric')
#gradient = gd5.create_gradient(metric='pnorm_condition_number')




objectPoints_des = pl.get_points()

# we now replace the first 4 points with the border positions
pl.uniform()
#objectPoints_des[:,0:4] = pl.get_points()


alpha=0.2
imagePoints_des = np.array(cam.project(objectPoints_des, False))
objectPoints_list = list()
imagePoints_list = list()
transfer_error_list = list()
condition_number_list = list()
normalized_condition_number_list = list()
new_objectPoints = objectPoints_des
for i in range(1000):
  objectPoints = np.copy(new_objectPoints)

  gradient = gd5.evaluate_gradient(gradient,objectPoints, np.array(cam.P))
  #gradient = gd5.normalize_gradient(gradient)

  new_objectPoints = gd5.update_points(alpha, gradient, objectPoints)#, limit = 3)
  new_imagePoints = np.array(cam.project(new_objectPoints, False))



  objectPoints_list.append(new_objectPoints)
  imagePoints_list.append(new_imagePoints)

  #plt.cla()
  plt.figure('Image Points')
  plt.ion()
  if i==0:
    plt.cla()
    cam.plot_plane(pl)
    plt.plot(imagePoints_des[0],imagePoints_des[1],'x',color = 'black',)
    plt.xlim(0,1280)
    plt.ylim(0,960)
    plt.gca().invert_yaxis()
    plt.axes().set_aspect('equal', 'datalim')

  plt.cla()
  cam.plot_plane(pl)
  plt.plot(new_imagePoints[0],new_imagePoints[1],'.',color = 'blue',)
  plt.pause(0.01)


  plt.figure('Object Points')
  plt.ion()
  if i==0:
    plt.cla()
    plt.plot(objectPoints_des[0],objectPoints_des[1],'x',color = 'black',)
    plt.axes().set_aspect('equal', 'datalim')

  plt.plot(new_objectPoints[0],new_objectPoints[1],'.',color = 'blue',)
  plt.pause(0.01)




  Xo = np.copy(new_objectPoints[[0,1,3],:]) #without the z coordinate (plane)
  Xi = np.copy(new_imagePoints)

  Hnoisy,A_t_ref,H_t = homo2d.homography2d(Xo,Xi)

  Aideal_norm = ef.calculate_A_matrix()

  x1,y1,x2,y2,x3,y3,x4,y4,x5,y5 = gd5.extract_objectpoints_vars(new_objectPoints)
  mat_cond_autrograd = gd5.matrix_condition_number_autograd(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,np.array(cam.P))


  volkerMetric = ef.volker_metric(Aideal)

  mat_cond = ef.get_matrix_pnorm_condition_number(Aideal)
  #mat_cond = get_matrix_conditioning_number(Aideal)

  condition_number_list.append()



  ##HOMOGRAPHY ERRORS
  ## TRUE VALUE OF HOMOGRAPHY OBTAINED FROM CAMERA PARAMETERS
  Hcam = cam.homography_from_Rt()
  homography_iters = 100
  ##We add noise to the image points and calculate the noisy homography
  transfer_error_sum = 0
  for j in range(homography_iters):
    new_imagePoints_noisy = cam.addnoise_imagePoints(new_imagePoints, mean = 0, sd = 2)
    #Noisy homography calculation
    Xo = new_objectPoints[[0,1,3],:]
    Xi = new_imagePoints_noisy
    Hnoisy,A_t_ref,H_t = homo2d.homography2d(Xo,Xi)
    Hnoisy = Hnoisy/Hnoisy[2,2]

    ## ERRORS FOR THE NOISY HOMOGRAPHY
    ## VALIDATION OBJECT POINTS
    validation_objectPoints =validation_plane.get_points()
    validation_imagePoints = np.array(cam.project(validation_objectPoints, False))
    Xo = np.copy(validation_objectPoints)
    Xo = np.delete(Xo, 2, axis=0)
    Xi = np.copy(validation_imagePoints)
    transfer_error_sum += ef.validation_points_error(Xi, Xo, Hnoisy)
  transfer_error_list.append(transfer_error_sum/homography_iters)

  plt.figure("Average Transfer error")
  plt.cla()
  plt.ion()
  plt.plot(transfer_error_list)
  plt.pause(0.01)





  print "Iteration: ", i
  print "Mat cond Autograd: ", mat_cond_autrograd
  print "Mat cond:", mat_cond
  print "Volker Metric:", volkerMetric
  print "dx1,dy1 :", gradient.dx1_eval,gradient.dy1_eval
  print "dx2,dy2 :", gradient.dx2_eval,gradient.dy2_eval
  print "dx3,dy3 :", gradient.dx3_eval,gradient.dy3_eval
  print "dx4,dy4 :", gradient.dx4_eval,gradient.dy4_eval
  print "dx5,dy5 :", gradient.dx5_eval,gradient.dy5_eval
  print "------------------------------------------------------"





plt.figure('Image Points')
plt.plot(new_imagePoints[0],new_imagePoints[1],'.',color = 'red',)
