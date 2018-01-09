#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:52:11 2017

@author: lracuna
"""
import autograd.numpy as np
from vision.rt_matrix import R_matrix_from_euler_t
from vision.camera import Camera
from vision.plane import Plane
from vision.circular_plane import CircularPlane
import matplotlib.pyplot as plt

import error_functions as ef
from ippe import homo2d
from homographyHarker.homographyHarker import homographyHarker as hh
from solve_ippe import pose_ippe_both, pose_ippe_best
from solve_pnp import pose_pnp
import cv2

number_of_points = 4

if number_of_points == 4:
    import gdescent.hpoints_gradient as gd
elif number_of_points == 5:
    import gdescent.hpoints_gradient5 as gd
elif number_of_points == 6:
    import gdescent.hpoints_gradient6 as gd



## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx = 800,fy = 800,cx = 640,cy = 480)
cam.set_width_heigth(960,960)

## DEFINE CAMERA POSE LOOKING STRAIGTH DOWN INTO THE PLANE MODEL
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
cam.set_t(0.0,-0.0,0.5, frame='world')

#cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(140.0))
#cam.set_t(0.0,-1,1.0, frame='world')
#
#r = 0.5
#angle = 10
#x = r*np.cos(np.deg2rad(angle))
#z = r*np.sin(np.deg2rad(angle))
#cam.set_t(0, x,z)
#cam.set_R_mat(R_matrix_from_euler_t(0.0,0,0))
#cam.look_at([0,0,0])

#cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(110.0))
#cam.set_t(0.0,-0.3,0.1, frame='world')

## Define a Display plane
pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.3,0.3), n = (2,2))
pl = CircularPlane()
pl.random(n =number_of_points, r = 0.01, min_sep = 0.01)


objectPoints = pl.get_points()

x1,y1,x2,y2,x3,y3,x4,y4 = gd.extract_objectpoints_vars(objectPoints)
imagePoints_true = np.array(cam.project(objectPoints, False))
imagePoints_measured = cam.addnoise_imagePoints(imagePoints_true, mean = 0, sd = 4)
repro =  gd.repro_error_autograd(x1,y1,x2,y2,x3,y3,x4,y4,cam.P, imagePoints_measured)
#%%




## CREATE A SET OF IMAGE POINTS FOR VALIDATION OF THE HOMOGRAPHY ESTIMATION
validation_plane =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.3,0.3), n = (4,4))
validation_plane.uniform()




## we create the gradient for the point distribution
normalize= False
n = 0.000000001 #condition number norm
gradient = gd.create_gradient(metric='condition_number', n = n)


#normalize= True
#n = 0.0001 #condition number norm
#gradient = gd.create_gradient(metric='condition_number', n = n)


#gradient = gd.create_gradient(metric='pnorm_condition_number')
#gradient = gd.create_gradient(metric='volker_metric')



objectPoints_des = pl.get_points()
# we now replace the first 4 points with the border positions
#pl.uniform()
#objectPoints_des[:,0:4] = pl.get_points()

#define the plots
#one Figure for image an object points
fig1 = plt.figure('Image and Object points')
ax_image = fig1.add_subplot(211)
ax_object = fig1.add_subplot(212)


#another figure for Homography error and condition numbers
fig2 = plt.figure('Error and condition numbers')
ax_error = plt.subplot(311)
ax_cond = plt.subplot(312, sharex = ax_error)
ax_cond_norm = plt.subplot(313, sharex = ax_error)



#another figure for Pose errors
fig3 = plt.figure('Pose estimation errors')
ax_t_error_ippe1 = fig3.add_subplot(421)
ax_r_error_ippe1 = fig3.add_subplot(422)
ax_t_error_ippe2 = fig3.add_subplot(423)
ax_r_error_ippe2 = fig3.add_subplot(424)
ax_t_error_epnp = fig3.add_subplot(425)
ax_r_error_epnp = fig3.add_subplot(426)
ax_t_error_pnp = fig3.add_subplot(427)
ax_r_error_pnp = fig3.add_subplot(4,2,8)


imagePoints_des = np.array(cam.project(objectPoints_des, False))
objectPoints_list = []
imagePoints_list = []
objectPoints_historic = np.array([]).reshape(objectPoints_des.shape[0],0)
transfer_error_list = []
cond_list = []
cond_norm_list = []
ippe_tvec_error_list1 = []
ippe_rmat_error_list1 = []
ippe_tvec_error_list2 = []
ippe_rmat_error_list2 = []
epnp_tvec_error_list = []
epnp_rmat_error_list = []
pnp_tvec_error_list = []
pnp_rmat_error_list = []

new_objectPoints = objectPoints_des
new_imagePoints = np.array(cam.project(new_objectPoints, False))
homography_iters = 1

for i in range(1000):
#  Xo = np.copy(new_objectPoints[[0,1,3],:]) #without the z coordinate (plane)
#  Xi = np.copy(new_imagePoints)
#  Aideal = ef.calculate_A_matrix(Xo, Xi)
  objectPoints_list.append(new_objectPoints)
  objectPoints_historic = np.hstack([objectPoints_historic,new_objectPoints])
  imagePoints_list.append(new_imagePoints)
#
#  input_list = gd.extract_objectpoints_vars(new_objectPoints)
#  input_list.append(np.array(cam.P))
#
#  mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize = False)
#
#  input_list = gd.extract_objectpoints_vars(new_objectPoints)
#  input_list.append(np.array(cam.P))
#  mat_cond_normalized = gd.matrix_condition_number_autograd(*input_list, normalize = True)
#
#  cond_list.append(mat_cond)
#  cond_norm_list.append(mat_cond_normalized)
#
#  ##HOMOGRAPHY ERRORS
#  ## TRUE VALUE OF HOMOGRAPHY OBTAINED FROM CAMERA PARAMETERS
#  Hcam = cam.homography_from_Rt()
#  ##We add noise to the image points and calculate the noisy homography
#  transfer_error_loop = []
#  ippe_tvec_error_loop1 = []
#  ippe_rmat_error_loop1 = []
#  ippe_tvec_error_loop2 = []
#  ippe_rmat_error_loop2 = []
#  epnp_tvec_error_loop = []
#  epnp_rmat_error_loop = []
#  pnp_tvec_error_loop = []
#  pnp_rmat_error_loop = []
#  for j in range(homography_iters):
#    new_imagePoints_noisy = cam.addnoise_imagePoints(new_imagePoints, mean = 0, sd = 1)
#
#
#    #Calculate the pose using IPPE (solution with least repro error)
#    normalizedimagePoints = cam.get_normalized_pixel_coordinates(new_imagePoints_noisy)
#    ippe_tvec1, ippe_rmat1, ippe_tvec2, ippe_rmat2 = pose_ippe_both(new_objectPoints, normalizedimagePoints, debug = False)
#    ippeCam1 = cam.clone_withPose(ippe_tvec1, ippe_rmat1)
#    ippeCam2 = cam.clone_withPose(ippe_tvec2, ippe_rmat2)
#
#    #Calculate the pose using solvepnp EPNP
#    epnp_tvec, epnp_rmat = pose_pnp(new_objectPoints, new_imagePoints_noisy, cam.K, debug, cv2.SOLVEPNP_EPNP,False)
#    epnpCam = cam.clone_withPose(epnp_tvec, epnp_rmat)
#
#    #Calculate the pose using solvepnp ITERATIVE
#    pnp_tvec, pnp_rmat = pose_pnp(new_objectPoints, new_imagePoints_noisy, cam.K, debug, cv2.SOLVEPNP_ITERATIVE,False)
#    pnpCam = cam.clone_withPose(pnp_tvec, pnp_rmat)
#
#    #Calculate errors
#    ippe_tvec_error1, ippe_rmat_error1 = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, ippeCam1.get_tvec(), ippe_rmat1)
#    ippe_tvec_error2, ippe_rmat_error2 = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, ippeCam2.get_tvec(), ippe_rmat2)
#    ippe_tvec_error_loop1.append(ippe_tvec_error1)
#    ippe_rmat_error_loop1.append(ippe_rmat_error1)
#    ippe_tvec_error_loop2.append(ippe_tvec_error2)
#    ippe_rmat_error_loop2.append(ippe_rmat_error2)
#
#    epnp_tvec_error, epnp_rmat_error = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, epnpCam.get_tvec(), epnp_rmat)
#    epnp_tvec_error_loop.append(epnp_tvec_error)
#    epnp_rmat_error_loop.append(epnp_rmat_error)
#
#    pnp_tvec_error, pnp_rmat_error = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, pnpCam.get_tvec(), pnp_rmat)
#    pnp_tvec_error_loop.append(pnp_tvec_error)
#    pnp_rmat_error_loop.append(pnp_rmat_error)
#
#
#
#
#
#    #Homography Estimation from noisy image points
#    Xo = new_objectPoints[[0,1,3],:]
#    Xi = new_imagePoints_noisy
#    #Hnoisy,A_t_ref,H_t = homo2d.homography2d(Xo,Xi)
#    #Hnoisy = Hnoisy/Hnoisy[2,2]
#    Hnoisy = hh(Xo,Xi)
#
#    ## ERRORS FOR THE NOISY HOMOGRAPHY
#    ## VALIDATION OBJECT POINTS
#    validation_objectPoints =validation_plane.get_points()
#    validation_imagePoints = np.array(cam.project(validation_objectPoints, False))
#    Xo = np.copy(validation_objectPoints)
#    Xo = np.delete(Xo, 2, axis=0)
#    Xi = np.copy(validation_imagePoints)
#    transfer_error_loop.append(ef.validation_points_error(Xi, Xo, Hnoisy))
#
#  transfer_error_list.append(np.mean(transfer_error_loop))
#  ippe_tvec_error_list1.append(np.mean(ippe_tvec_error_loop1))
#  ippe_rmat_error_list1.append(np.mean(ippe_rmat_error_loop1))
#  ippe_tvec_error_list2.append(np.mean(ippe_tvec_error_loop2))
#  ippe_rmat_error_list2.append(np.mean(ippe_rmat_error_loop2))
#  epnp_tvec_error_list.append(np.mean(epnp_tvec_error_loop))
#  epnp_rmat_error_list.append(np.mean(epnp_rmat_error_loop))
#  pnp_tvec_error_list.append(np.mean(pnp_tvec_error_loop))
#  pnp_rmat_error_list.append(np.mean(pnp_rmat_error_loop))
#
#
  #PLOT IMAGE POINTS
  plt.sca(ax_image)
  plt.ion()
  if i==0:
    plt.cla()
    cam.plot_plane(pl)
    plt.plot(imagePoints_des[0],imagePoints_des[1],'x',color = 'black',)
    ax_image.set_aspect('equal', 'datalim')
  ax_image.cla()
  cam.plot_plane(pl)
  ax_image.plot(new_imagePoints[0],new_imagePoints[1],'.',color = 'blue',)

  ax_image.invert_yaxis()
  ax_image.set_xlim(0,1280)
  ax_image.set_ylim(0,960)
  ax_image.set_title('Image Points')
#
#
  #PLOT OBJECT POINTS
  plt.sca(ax_object)
  if i==0:
    ax_object.cla()
    ax_object.plot(objectPoints_des[0],objectPoints_des[1],'x',color = 'black',)
    ax_object.set_aspect('equal', 'datalim')
  ax_object.cla()
  plt.ion()
  ax_object.plot(objectPoints_historic[0],objectPoints_historic[1],'.',color = 'blue',)
  ax_object.plot(objectPoints_historic[0,-number_of_points:],objectPoints_historic[1,-number_of_points:],'.',color = 'red',)
  pl.plot_plane()
  ax_object.set_title('Object Points')
  plt.show()
  plt.pause(0.001)
#
#  #PLOT TRANSFER ERROR
#  plt.sca(ax_error)
#  plt.ion()
#  ax_error.cla()
#  ax_error.plot(transfer_error_list)
#
#
#  #PLOT CONDITION NUMBER
#  plt.sca(ax_cond)
#  plt.ion()
#  ax_cond.cla()
#  ax_cond.plot(cond_list)
#
#  #PLOT CONDITION NUMBER NORMALIZED
#  plt.sca(ax_cond_norm)
#  plt.ion()
#  ax_cond_norm.cla()
#  ax_cond_norm.plot(cond_norm_list)
#
#  plt.setp(ax_error.get_xticklabels(), visible=False)
#  plt.setp(ax_cond.get_xticklabels(), visible=False)
#
#  ax_error.set_title('Geometric Transfer error of the validation points')
#  ax_cond.set_title('Condition number of the A matrix')
#  ax_cond_norm.set_title('Condition number of the Normalized A matrix')
#
#  plt.show()
#  plt.pause(0.001)
#
#
#  #PLOT POSE ERRORS
#  #IPPE
#  plt.sca(ax_t_error_ippe1)
#  plt.ion()
#  ax_t_error_ippe1.cla()
#  ax_t_error_ippe1.plot(ippe_tvec_error_list1)
#
#  plt.sca(ax_r_error_ippe1)
#  plt.ion()
#  ax_r_error_ippe1.cla()
#  ax_r_error_ippe1.plot(ippe_rmat_error_list1)
#
#  plt.sca(ax_t_error_ippe2)
#  plt.ion()
#  ax_t_error_ippe2.cla()
#  ax_t_error_ippe2.plot(ippe_tvec_error_list2)
#
#  plt.sca(ax_r_error_ippe2)
#  plt.ion()
#  ax_r_error_ippe2.cla()
#  ax_r_error_ippe2.plot(ippe_rmat_error_list2)
#
#  #EPNP
#  plt.sca(ax_t_error_epnp)
#  plt.ion()
#  ax_t_error_epnp.cla()
#  ax_t_error_epnp.plot(epnp_tvec_error_list)
#
#  plt.sca(ax_r_error_epnp)
#  plt.ion()
#  ax_r_error_epnp.cla()
#  ax_r_error_epnp.plot(epnp_rmat_error_list)
#
#  #ITERATIVE PNP
#  plt.sca(ax_t_error_pnp)
#  plt.ion()
#  ax_t_error_pnp.cla()
#  ax_t_error_pnp.plot(pnp_tvec_error_list)
#
#  plt.sca(ax_r_error_pnp)
#  plt.ion()
#  ax_r_error_pnp.cla()
#  ax_r_error_pnp.plot(pnp_rmat_error_list)
#
#
#
#  ax_t_error_ippe1.set_title('Translation error (in percent) for IPPE Pose 1')
#  ax_r_error_ippe1.set_title('Rotation error (Angle) for IPPE Pose 1')
#  ax_t_error_ippe2.set_title('Translation error (in percent) for IPPE Pose 2')
#  ax_r_error_ippe2.set_title('Rotation error (Angle) for IPPE Pose 2')
#  ax_t_error_dls.set_title('Translation error (in percent) for DLS Pose')
#  ax_r_error_dls.set_title('Rotation error (Angle) for DLS Pose')
#  ax_t_error_epnp.set_title('Translation error (in percent) for EPnP Pose')
#  ax_r_error_epnp.set_title('Rotation error (Angle) for EPnP Pose')
#  ax_t_error_pnp.set_title('Translation error (in percent) for PnP Pose')
#  ax_r_error_pnp.set_title('Rotation error (Angle) for PnP Pose')
#
#  plt.show()
#  plt.pause(0.001)
#
#
#
#
  print "Iteration: ", i
#  print "Transfer Error: ", np.mean(transfer_error_loop)
#  print "Mat cond:", mat_cond
#  print "Mat cond normalized:", mat_cond_normalized
#  print "Points", new_objectPoints
#  print "dx1,dy1 :", gradient.dx1_eval,gradient.dy1_eval
#  print "dx2,dy2 :", gradient.dx2_eval,gradient.dy2_eval
#  print "dx3,dy3 :", gradient.dx3_eval,gradient.dy3_eval
#  print "dx4,dy4 :", gradient.dx4_eval,gradient.dy4_eval
#  print "------------------------------------------------------"

  ## GRADIENT DESCENT

  objectPoints = np.copy(new_objectPoints)
  gradient = gd.evaluate_gradient(gradient,objectPoints, np.array(cam.P), normalize)

  new_objectPoints = gd.update_points(gradient, objectPoints, limitx=0.15,limity=0.15)
  new_imagePoints = np.array(cam.project(new_objectPoints, False))

#plt.figure('Image Points')
#plt.plot(new_imagePoints[0],new_imagePoints[1],'.',color = 'red',)


