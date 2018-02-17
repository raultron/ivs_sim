#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:52:11 2017

@author: lracuna
"""
import autograd.numpy as np
import pickle
import sys
sys.path.append("../vision/")
sys.path.append("../gdescent/")
from vision.rt_matrix import R_matrix_from_euler_t
from vision.camera import Camera
from vision.plane import Plane
from vision.circular_plane import CircularPlane
import error_functions as ef
from ippe import homo2d
from homographyHarker.homographyHarker import homographyHarker as hh
from solve_ippe import pose_ippe_both, pose_ippe_best
from solve_pnp import pose_pnp
import cv2
import matplotlib.pyplot as plt

calc_metrics = True
number_of_points = 4

if number_of_points == 4:
    import gdescent.hpoints_gradient as gd
elif number_of_points == 5:
    import gdescent.hpoints_gradient5 as gd
elif number_of_points == 6:
    import gdescent.hpoints_gradient6 as gd

## Define a Display plane with random initial points
pl = CircularPlane()
pl.random(n =number_of_points, r = 0.01, min_sep = 0.01)

## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx = 800,fy = 800,cx = 640/2.,cy = 480/2.)
cam.set_width_heigth(640,480)

## DEFINE CAMERA POSE LOOKING STRAIGTH DOWN INTO THE PLANE MODEL
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
cam.set_t(0.0,-0.0,0.5, frame='world')

#cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(140.0))
#cam.set_t(0.0,-1,1.0, frame='world')
#
r = 0.8
angle = 90
x = r*np.cos(np.deg2rad(angle))
z = r*np.sin(np.deg2rad(angle))
cam.set_t(0, x,z)
cam.set_R_mat(R_matrix_from_euler_t(0.0,0,0))
cam.look_at([0,0,0])

#cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(110.0))
#cam.set_t(0.0,-0.3,0.1, frame='world')


new_objectPoints = pl.get_points()
#GOOD SET OF POINTS FOR THE GRAPHIC
#new_objectPoints = np.array([[ 0.075, -0.06,   0.06,  -0.06 ],
# [ 0.105,  0.105,  0.105,  0.09 ],
# [ 0.,     0.,     0.,     0.,   ],
# [ 1.,     1.,     1.,     1.,   ]])

new_objectPoints = np.array([[ 0.12,   0.06,  -0.105,  0.105],
 [ 0.015, -0.12,  -0.09,  -0.09 ],
 [ 0.,     0.,     0.,     0.,   ],
 [ 1.,     1.,     1.,     1.,   ]])


## CREATE A SET OF IMAGE POINTS FOR VALIDATION OF THE HOMOGRAPHY ESTIMATION
# This will create a grid of 16 points of size = (0.3,0.3) meters
validation_plane =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.3,0.3), n = (4,4))
validation_plane.uniform()

## we create the gradient for the point distribution
normalize= False
n = 0.00000005 #condition number norm 4 points
gradient = gd.create_gradient(metric='condition_number', n = n)


#define the plots
#one Figure for image and object points
fig11 = plt.figure('Image Plane Coordinates')
ax_image = fig11.add_subplot(111)
fig12 = plt.figure('Control Points plane Coordinates')
ax_object = fig12.add_subplot(111)

if calc_metrics:
    #another figure for Homography error and condition numbers
    fig2 = plt.figure('Effect of point configuration in homography estimation')
    ax_cond = plt.subplot(211)    
    ax_homo_error = plt.subplot(212, sharex = ax_cond)    
    
    
    #another figure for Pose errors
    fig3 = plt.figure('Effect of point configuration in Pose estimation')
    
    ax_t_error_ippe = fig3.add_subplot(321)
    ax_r_error_ippe = fig3.add_subplot(322)
    ax_t_error_epnp = fig3.add_subplot(323, sharex = ax_t_error_ippe)
    ax_r_error_epnp = fig3.add_subplot(324, sharex = ax_r_error_ippe)    
    ax_t_error_pnp = fig3.add_subplot(325, sharex = ax_t_error_ippe)
    ax_r_error_pnp = fig3.add_subplot(3,2,6, sharex = ax_r_error_ippe)


#Dictionary with all the important information for one run
DataOut = {}
DataOut['n'] = number_of_points
DataOut['Camera'] = cam.clone()
DataOut['Plane'] = pl
DataOut['ObjectPoints'] = []
DataOut['ImagePoints'] = []
DataOut['cond_number'] = []
DataOut['cond_number_norm'] = []
#For each iteration we will have a list of size homography_iters inside the main list
DataOut['homo_dlt_error'] = []
DataOut['homo_HO_error'] = []
DataOut['homo_CV_error'] = []
DataOut['ippe_tvec_error'] = []
DataOut['ippe_rmat_error'] = []
DataOut['epnp_tvec_error'] = []
DataOut['epnp_rmat_error'] = []
DataOut['pnp_tvec_error'] = []
DataOut['pnp_rmat_error'] = []

homo_dlt_error_mean = []
homo_HO_error_mean = []
homo_CV_error_mean = []
ippe_tvec_error_mean = []
ippe_rmat_error_mean = []
epnp_tvec_error_mean = []
epnp_rmat_error_mean = []
pnp_tvec_error_mean = []
pnp_rmat_error_mean = []

objectPoints_historic = np.array([]).reshape(new_objectPoints.shape[0],0)
new_imagePoints = np.array(cam.project(new_objectPoints, False))
homography_iters = 1000

for i in range(40):
  DataOut['ObjectPoints'].append(new_objectPoints)
  DataOut['ImagePoints'].append(new_imagePoints)
  objectPoints_historic = np.hstack([objectPoints_historic,new_objectPoints])
 
  
  #PLOT IMAGE POINTS
  plt.sca(ax_image)  
  ax_image.cla()
  cam.plot_plane(pl)
  ax_image.plot(new_imagePoints[0],new_imagePoints[1],'.',color = 'blue',)

  ax_image.set_xlim(0,cam.img_width)
  ax_image.set_ylim(0,cam.img_height)
  ax_image.invert_yaxis()
  ax_image.set_aspect('equal')
  ax_image.set_title('Image Points')

  #PLOT OBJECT POINTS
  plt.sca(ax_object)
  ax_object.cla()
  ax_object.plot(objectPoints_historic[0,0::4],objectPoints_historic[1,0::4],'-x',color = 'blue',)
  ax_object.plot(objectPoints_historic[0,1::4],objectPoints_historic[1,1::4],'-x',color = 'orange',)
  ax_object.plot(objectPoints_historic[0,2::4],objectPoints_historic[1,2::4],'-x',color = 'black',)
  ax_object.plot(objectPoints_historic[0,3::4],objectPoints_historic[1,3::4],'-x',color = 'magenta',)
  ax_object.plot(objectPoints_historic[0,-number_of_points:],objectPoints_historic[1,-number_of_points:],'.',color = 'red',)
  pl.plot_plane()
  ax_object.set_title('Object Points')
  ax_object.set_xlim(-pl.radius-0.05,pl.radius+0.05)
  ax_object.set_ylim(-pl.radius-0.05,pl.radius+0.05)
  ax_object.set_aspect('equal')
  plt.show()
  plt.pause(0.001)
  
  if calc_metrics:
      #CONDITION NUMBER CALCULATION
      input_list = gd.extract_objectpoints_vars(new_objectPoints)
      input_list.append(np.array(cam.P))    
      mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize = False)
     
      #CONDITION NUMBER WITH A NORMALIZED CALCULATION
      input_list = gd.extract_objectpoints_vars(new_objectPoints)
      input_list.append(np.array(cam.P))
      mat_cond_normalized = gd.matrix_condition_number_autograd(*input_list, normalize = True)
    
      DataOut['cond_number'].append(mat_cond)
      DataOut['cond_number_norm'].append(mat_cond_normalized)
    
      ##HOMOGRAPHY ERRORS
      ## TRUE VALUE OF HOMOGRAPHY OBTAINED FROM CAMERA PARAMETERS
      Hcam = cam.homography_from_Rt()
      ##We add noise to the image points and calculate the noisy homography
      homo_dlt_error_loop = []
      homo_HO_error_loop = []
      homo_CV_error_loop = []
      ippe_tvec_error_loop = []
      ippe_rmat_error_loop = []
      epnp_tvec_error_loop = []
      epnp_rmat_error_loop = []
      pnp_tvec_error_loop = []
      pnp_rmat_error_loop = []
      
      ImageNoise = 4
      DataOut['ImageNoise'] = ImageNoise
      
      # WE CREATE NOISY IMAGE POINTS (BASED ON THE TRUE VALUES) AND CALCULATE
      # THE ERRORS WE THEN OBTAIN AN AVERAGE FOR EACH ONE
      for j in range(homography_iters):
        new_imagePoints_noisy = cam.addnoise_imagePoints(new_imagePoints, mean = 0, sd = ImageNoise)
    
    
        #Calculate the pose using IPPE (solution with least repro error)
        normalizedimagePoints = cam.get_normalized_pixel_coordinates(new_imagePoints_noisy)
        ippe_tvec1, ippe_rmat1, ippe_tvec2, ippe_rmat2 = pose_ippe_both(new_objectPoints, normalizedimagePoints, debug = False)
        ippeCam1 = cam.clone_withPose(ippe_tvec1, ippe_rmat1)
        ippeCam2 = cam.clone_withPose(ippe_tvec2, ippe_rmat2)
    
        #Calculate the pose using solvepnp EPNP
        debug = False
        epnp_tvec, epnp_rmat = pose_pnp(new_objectPoints, new_imagePoints_noisy, cam.K, debug, cv2.SOLVEPNP_EPNP,False)
        epnpCam = cam.clone_withPose(epnp_tvec, epnp_rmat)
    
        #Calculate the pose using solvepnp ITERATIVE
        pnp_tvec, pnp_rmat = pose_pnp(new_objectPoints, new_imagePoints_noisy, cam.K, debug, cv2.SOLVEPNP_ITERATIVE,False)
        pnpCam = cam.clone_withPose(pnp_tvec, pnp_rmat)
    
        #Calculate errors
        ippe_tvec_error1, ippe_rmat_error1 = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, ippeCam1.get_tvec(), ippe_rmat1)
        ippe_tvec_error_loop.append(ippe_tvec_error1)
        ippe_rmat_error_loop.append(ippe_rmat_error1)        
    
        epnp_tvec_error, epnp_rmat_error = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, epnpCam.get_tvec(), epnp_rmat)
        epnp_tvec_error_loop.append(epnp_tvec_error)
        epnp_rmat_error_loop.append(epnp_rmat_error)
    
        pnp_tvec_error, pnp_rmat_error = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, pnpCam.get_tvec(), pnp_rmat)
        pnp_tvec_error_loop.append(pnp_tvec_error)
        pnp_rmat_error_loop.append(pnp_rmat_error) 
    
        #Homography Estimation from noisy image points
    
        #DLT TRANSFORM
        Xo = new_objectPoints[[0,1,3],:]
        Xi = new_imagePoints_noisy
        Hnoisy_dlt,_,_ = homo2d.homography2d(Xo,Xi)
        Hnoisy_dlt = Hnoisy_dlt/Hnoisy_dlt[2,2]
        
        #HO METHOD
        Xo = new_objectPoints[[0,1,3],:]
        Xi = new_imagePoints_noisy
        Hnoisy_HO = hh(Xo,Xi)
        
        #OpenCV METHOD
        Xo = new_objectPoints[[0,1,3],:]
        Xi = new_imagePoints_noisy
        Hnoisy_OpenCV,_ = cv2.findHomography(Xo[:2].T.reshape(1,-1,2),Xi[:2].T.reshape(1,-1,2))
        
    
        ## ERRORS FOR THE  DLT HOMOGRAPHY 
        ## VALIDATION OBJECT POINTS
        validation_objectPoints =validation_plane.get_points()
        validation_imagePoints = np.array(cam.project(validation_objectPoints, False))
        Xo = np.copy(validation_objectPoints)
        Xo = np.delete(Xo, 2, axis=0)
        Xi = np.copy(validation_imagePoints)
        homo_dlt_error_loop.append(ef.validation_points_error(Xi, Xo, Hnoisy_dlt))
        
        ## ERRORS FOR THE  HO HOMOGRAPHY 
        ## VALIDATION OBJECT POINTS
        validation_objectPoints =validation_plane.get_points()
        validation_imagePoints = np.array(cam.project(validation_objectPoints, False))
        Xo = np.copy(validation_objectPoints)
        Xo = np.delete(Xo, 2, axis=0)
        Xi = np.copy(validation_imagePoints)        
        homo_HO_error_loop.append(ef.validation_points_error(Xi, Xo, Hnoisy_HO))
        
        ## ERRORS FOR THE  OpenCV HOMOGRAPHY 
        ## VALIDATION OBJECT POINTS
        validation_objectPoints =validation_plane.get_points()
        validation_imagePoints = np.array(cam.project(validation_objectPoints, False))
        Xo = np.copy(validation_objectPoints)
        Xo = np.delete(Xo, 2, axis=0)
        Xi = np.copy(validation_imagePoints)        
        homo_CV_error_loop.append(ef.validation_points_error(Xi, Xo, Hnoisy_OpenCV))
    
      homo_dlt_error_mean.append(np.mean(homo_dlt_error_loop))
      homo_HO_error_mean.append(np.mean(homo_HO_error_loop))
      homo_CV_error_mean.append(np.mean(homo_CV_error_loop))
      ippe_tvec_error_mean.append(np.mean(ippe_tvec_error_loop))
      ippe_rmat_error_mean.append(np.mean(ippe_rmat_error_loop))     
      epnp_tvec_error_mean.append(np.mean(epnp_tvec_error_loop))
      epnp_rmat_error_mean.append(np.mean(epnp_rmat_error_loop))
      pnp_tvec_error_mean.append(np.mean(pnp_tvec_error_loop))
      pnp_rmat_error_mean.append(np.mean(pnp_rmat_error_loop))
      
      DataOut['homo_dlt_error'].append(homo_dlt_error_loop)
      DataOut['homo_HO_error'].append(homo_HO_error_loop)
      DataOut['homo_CV_error'].append(homo_CV_error_loop)
      
      DataOut['ippe_tvec_error'].append(ippe_tvec_error_loop)
      DataOut['ippe_rmat_error'].append(ippe_rmat_error_loop)
      
      DataOut['epnp_tvec_error'].append(epnp_tvec_error_loop)
      DataOut['epnp_rmat_error'].append(epnp_rmat_error_loop)
      
      DataOut['pnp_tvec_error'].append(pnp_tvec_error_loop)
      DataOut['pnp_rmat_error'].append(pnp_rmat_error_loop)   
        

      #################################
      #PLOT CONDITION NUMBER
      plt.sca(ax_cond)
      plt.ion()
      ax_cond.cla()
      ax_cond.plot(DataOut['cond_number'])
#      #PLOT CONDITION NUMBER NORMALIZED
#      plt.sca(ax_cond_norm)
#      plt.ion()
#      ax_cond_norm.cla()
#      ax_cond_norm.plot(cond_norm_list)
      
      #PLOT HOMO DLT ERROR
      plt.sca(ax_homo_error)
      plt.ion()
      ax_homo_error.cla()
      ax_homo_error.plot(homo_dlt_error_mean)      
      #PLOT HOMO HO ERROR
      ax_homo_error.plot(homo_HO_error_mean)      
#      ax_cond_norm.set_title('Condition number of the Normalized A matrix')   
      plt.show()
      plt.pause(0.001)
    
      ##################################
      #PLOT POSE ERRORS
      #IPPE
      plt.sca(ax_t_error_ippe)
      plt.ion()
      ax_t_error_ippe.cla()      
      ax_t_error_ippe.plot(ippe_tvec_error_mean,'-x') #semilogy
      ax_t_error_ippe.set_ylim(0,20)
    
      plt.sca(ax_r_error_ippe)
      plt.ion()
      ax_r_error_ippe.cla()
      ax_r_error_ippe.plot(ippe_rmat_error_mean,'-x')
       
      #EPNP
      plt.sca(ax_t_error_epnp)
      plt.ion()
      ax_t_error_epnp.cla()
      ax_t_error_ippe.plot(epnp_tvec_error_mean,'-x')
      
      plt.sca(ax_r_error_epnp)
      plt.ion()
      ax_r_error_epnp.cla()
      ax_r_error_epnp.plot(epnp_rmat_error_mean,'-x')
    
      #ITERATIVE PNP
      plt.sca(ax_t_error_pnp)
      plt.ion()
      ax_t_error_pnp.cla()
      ax_t_error_ippe.plot(pnp_tvec_error_mean,'-x')
    
      plt.sca(ax_r_error_pnp)
      plt.ion()
      ax_r_error_pnp.cla()
      ax_r_error_pnp.plot(pnp_rmat_error_mean,'-x')    
    
      #Figure Titles      
      
      ax_cond.set_title('Condition number of the A matrix')
      ax_homo_error.set_title('Geometric Transfer error of the validation points')
      ax_t_error_ippe.set_title('Translation error (in percent) for IPPE Pose 1')
      ax_r_error_ippe.set_title('Rotation error (Angle) for IPPE Pose 1')
      ax_t_error_epnp.set_title('Translation error (in percent) for EPnP Pose')
      ax_r_error_epnp.set_title('Rotation error (Angle) for EPnP Pose')
      ax_t_error_pnp.set_title('Translation error (in percent) for PnP Pose')
      ax_r_error_pnp.set_title('Rotation error (Angle) for PnP Pose')      
      
      plt.setp(ax_homo_error.get_xticklabels(), visible=False)
      plt.setp(ax_cond.get_xticklabels(), visible=False)      
      plt.setp(ax_t_error_ippe.get_xticklabels(), visible=False)
      plt.setp(ax_r_error_ippe.get_xticklabels(), visible=False)      
      plt.setp(ax_t_error_epnp.get_xticklabels(), visible=False)
      plt.setp(ax_r_error_epnp.get_xticklabels(), visible=False)
      
      plt.tight_layout()    
      plt.show()
      plt.pause(0.001)
    
    
      print "Iteration: ", i
      print "Transfer Error: ", homo_dlt_error_mean[-1]
      print "Mat cond:", mat_cond
      print "Mat cond normalized:", mat_cond_normalized
      print "Points", new_objectPoints
      print "dx1,dy1 :", gradient.dx1_eval,gradient.dy1_eval
      print "dx2,dy2 :", gradient.dx2_eval,gradient.dy2_eval
      print "dx3,dy3 :", gradient.dx3_eval,gradient.dy3_eval
      print "dx4,dy4 :", gradient.dx4_eval,gradient.dy4_eval
      print "------------------------------------------------------"
  print "Iteration: ", i
  
  ## GRADIENT DESCENT
  objectPoints = np.copy(new_objectPoints)
  #ASSUMING NOISy MEARUREMENST
  new_imagePoints_noisy = cam.addnoise_imagePoints(new_imagePoints, mean = 0, sd =4)
  #Calculate the pose using solvepnp ITERATIVE
  pnp_tvec_test, pnp_rmat_test = pose_pnp(objectPoints, new_imagePoints_noisy, cam.K, False, cv2.SOLVEPNP_ITERATIVE,False)
  pnpCam_test = cam.clone_withPose(pnp_tvec_test, pnp_rmat_test)
  pnpCam_test.update_Rt()  
  gradient = gd.evaluate_gradient(gradient,objectPoints, np.array(pnpCam_test.P), normalize)
  #gradient = gd.evaluate_gradient(gradient,objectPoints, np.array(cam.P), normalize)
  new_objectPoints = gd.update_points(gradient, objectPoints, limitx=0.15,limity=0.15)
  new_imagePoints = np.array(cam.project(new_objectPoints, False))

pickle.dump( DataOut, open( "icra_sim1_frontoparallel_noisy_cond_number.p", "wb" ) )
