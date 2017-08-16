# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:30:20 2017

@author: racuna
"""
"""TODO:
1) Move error functions to the centralized error_functions.py. Easier to detect errors
"""
from pose_sim import *
from vision.camera import *
from vision.plane import Plane
from vision.screen import Screen
from vision.camera_distribution import create_cam_distribution
from ippe import homo2d

import numpy as np
import matplotlib.pyplot as plt
import error_functions as ef


#############################################
## INITIALIZATIONS
#############################################

## CREATE A SIMULATED CAMERA
cam = Camera()
cam.set_K(fx=800,fy=800,cx=640,cy=480)
cam.set_width_heigth(1280,960)
## DEFINE CAMERA POSE LOOKING STRAIGTH DOWN INTO THE PLANE MODEL
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(160.0))
cam.set_t(0.0,-0.2,0.4, frame = 'world')
cam.set_P()

#Create a plane with 4 points to start
pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.2,0.2), n = (2,2))
pl.uniform()

cams = create_cam_distribution(cam, pl, 0.01, plot=False)

## OBTAIN REFERENCE MODEL POINTS (A PERFECT SQUARE)
objectPoints_ref = pl.get_points()
#objectPoints_ref = np.load('objectPoints_test_ippe5.npy')

## CREATE A SET OF OBJECT POINTS EVENLY DISTRIBUTED FOR VALIDATION OF THE HOMOGRAPHY ESTIMATION
validation_plane =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.4,0.4), n = (4,4))
validation_plane.uniform()


#############################################
## WE START THE MAIN LOOP OF CALCULATIONS
#############################################

error_ref = list()
error_desired = list()
transfer_error_ref = list()
transfer_error_des = list()
volker_metric_ref = list()
volker_metric_des = list()
matrix_cond_ref = list()
matrix_cond_des = list()
algebraic_error_ref = list()
algebraic_error_des = list()

max_iters = 10

for i in range(max_iters):
  for cam in cams:
    ## TRUE VALUE OF HOMOGRAPHY OBTAINED FROM CAMERA PARAMETERS
    Hcam = cam.homography_from_Rt()   
    
    ## GENERATION OF THE DESIRED OBJECT POINTS    
    #pl.uniform_with_distortion(mean = 0, sd = 0.02)
    pl.random(n = 20, r = 0.01, min_sep = 0.01)
    objectPoints_des = pl.get_points()
    
    ## PROJECT OBJECT POINTS INTO CAMERA IMAGE
    imagePoints_ref = np.array(cam.project(objectPoints_ref, False))
    imagePoints_des = np.array(cam.project(objectPoints_des, False))

    ## ADD NOISE
    imagePoints_ref_noisy = cam.addnoise_imagePoints(imagePoints_ref, mean = 0, sd = 2)
    imagePoints_des_noisy = cam.addnoise_imagePoints(imagePoints_des, mean = 0, sd = 2)

    ## SHOW THE PROJECTIONS IN IMAGE PLANE
    if i % 10 == 0:
        plt.ion()
        plt.figure('Image Points')
        plt.plot(imagePoints_ref_noisy[0],imagePoints_ref_noisy[1],'.',color = 'black',  label='Reference',)
        plt.plot(imagePoints_des_noisy[0],imagePoints_des_noisy[1],'x',color = 'r',  label='Desired',)
        plt.xlim(0,1280)
        plt.ylim(0,960)
        #if i==0:
        #    plt.legend()

        plt.gca().invert_yaxis()
    #plt.pause(0.05)

    ## NOW FOR EACH SET OF POINTS WE CALCULATE THE HOMOGRAPHY USING THE DLT ALGORITHM

    ## REFERENCE OBJECT POINTS
    Xo = objectPoints_ref[[0,1,3],:]
    Xi = imagePoints_ref_noisy
    Aref = ef.calculate_A_matrix(Xo, Xi)
    Href,A_t_ref,H_t = homo2d.homography2d(Xo,Xi)
    Href = Href/Href[2,2]  
    
    ## NOW FOR POINTS WITH A DESIRED CONFIGURATION
    Xo = np.copy(objectPoints_des)
    Xo = np.delete(Xo, 2, axis=0)
    Xi = np.copy(imagePoints_des_noisy)
    Ades = ef.calculate_A_matrix(Xo, Xi)
    Hdes,A_t_des,_ = homo2d.homography2d(Xo,Xi)
    Hdes = Hdes/Hdes[2,2]


    ## CALCULATE HOMOGRAPHY ESTIMATION ERRORS
    ## we need points without noise to confirm
    ## object coordinates dont have noise, only image coordinates

    ## VALIDATION OBJECT POINTS

    validation_objectPoints =validation_plane.get_points()
    validation_imagePoints = np.array(cam.project(validation_objectPoints, False))
    Xo = np.copy(validation_objectPoints)
    Xo = np.delete(Xo, 2, axis=0)
    Xi = np.copy(validation_imagePoints)      
    
    
    ## ERRORS FOR THE REFERENCE HOMOGRAPHY
    transfer_error = ef.validation_points_error(Xi, Xo, Href)
    transfer_error_ref.append(transfer_error)

    H_error = ef.homography_matrix_error(Hcam, Href)
    error_ref.append(H_error)
    
    ## ERRORS FOR THE DESIRED HOMOGRAPHY    
    transfer_error = ef.validation_points_error(Xi, Xo, Hdes)
    transfer_error_des.append(transfer_error)

    H_error = ef.homography_matrix_error(Hcam, Hdes)
    error_desired.append(H_error)


    volker_metric_ref.append(ef.volker_metric(Aref))
    volker_metric_des.append(ef.volker_metric(Ades))

    ## This metric is based on the condition number (2-norm)
    ## Explained in chapter 5 section 2.2: http://www.math.ucla.edu/~dakuang/cse6040/lectures/6040_lecture15.pdf
    #More information about matrix perturbation theory
    #https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-241j-dynamic-systems-and-control-spring-2011/readings/MIT6_241JS11_chap05.pdf

    matrix_cond_ref.append(ef.get_matrix_pnorm_condition_number(A_t_ref))
    matrix_cond_des.append(ef.get_matrix_pnorm_condition_number(A_t_des))

    algebraic_error_ref.append(np.linalg.norm(np.dot(Aref,Hcam.ravel().T)))
    algebraic_error_des.append(np.linalg.norm(np.dot(Ades,Hcam.ravel().T)))

error_ref = np.array(error_ref)
error_desired = np.array(error_desired)
comp = error_ref < error_desired
print "Times reference pattern was better than desired pattern: ", np.count_nonzero(comp)
print "--------------------------------------------------------"
print "H Matrix absolute Error"
print "Ref: ", np.mean(error_ref)
print "Des: ", np.mean(error_desired)
print "--------------------------------------------------------"

print "Geometric distance for the validation points"
print "Ref: ", np.mean(transfer_error_ref)
print "Des: ", np.mean(transfer_error_des)
print "--------------------------------------------------------"

print "Matrix conditioning"
print "Ref: ", np.mean(matrix_cond_ref)
print "Des: ", np.mean(matrix_cond_des)
print "--------------------------------------------------------"

print "Algebraic Error of the A matrix and the estimated H. norm(dot(A,H))"
print "Ref: ", np.mean(algebraic_error_ref)
print "Des: ", np.mean(algebraic_error_des)
print "--------------------------------------------------------"

print "Volker Metric"
print 'Ref: ', np.mean(volker_metric_ref)
print 'Des: ', np.mean(volker_metric_des)

""" MAYBE DELETE """
#
##Example 2: Ref better than desired in all metrics except the volker_metric
#dx = 0
#dy = 0
#x = np.linspace(185+dx,805+dx-400,2)
#y = np.linspace(235+dy,752+dy-300,2)

##Example 3: Ref better than desired in all metrics except the volker_metric
#dx = -100
#dy = -100
#x = np.linspace(185+dx,805+dx-400,2)
#y = np.linspace(235+dy,752+dy-300,2)

##Example 4: Ref worst than desired in all metrics INCLUDING the volker_metric
#dx = -100
#dy = -100
#x = np.linspace(185+dx,805+dx,2)
#y = np.linspace(235+dy,752+dy,2)

##Example 5: Ref worst than desired in all metrics except the volker_metric
#dx = -100
#dy = -100
#x = np.linspace(185+dx,1200+dx,2)
#y = np.linspace(235+dy,900+dy,2)

#Example 5: Ref worst than desired in all metrics except the volker_metric
#dx = 100
#dy = -10
#x = np.linspace(350+dx,952+dx,2)*0.5#952
#y = np.linspace(107+dy,771+dy,2)*0.9
#xx,yy = np.meshgrid(x,y)
#desired_imageCoordinates = np.array([xx.ravel(),yy.ravel(), np.ones_like(yy.ravel())])
#
#objectPoints_des = np.dot(np.linalg.inv(H_cam),desired_imageCoordinates)
#objectPoints_des = np.insert(objectPoints_des,2,np.zeros(objectPoints_des.shape[1]), axis = 0)
#
#pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.2,0.2), n = (4,4))
#pl.uniform_with_distortion(mean = 0, sd = 0.01)
##pl.uniform()
#objectPoints_des = pl.get_points()
#
###NORMALIZE COORDINATES AND FIX THE Z TO ZERO (ALL POINTS ON A PLANE ON 3D COORDINATES)
#for i in range(objectPoints_des.shape[1]):
#  objectPoints_des[0,i] = objectPoints_des[0,i]/objectPoints_des[3,i]
#  objectPoints_des[1,i] = objectPoints_des[1,i]/objectPoints_des[3,i]
#  objectPoints_des[2,i] = 0
#  objectPoints_des[3,i] = 1

#pl.uniform_with_distortion(mean = 0, sd = 0.02)
#objectPoints_des = pl.get_points()

### CREATE AN IDEAL SET OF POINTS IN IMAGE COORDINATES (A SQUARE)
### FROM THE DESIRED IMAGE POINTS AND THE CAMERA MATRIX OBTAIN THE COORDINATES IN
### MODEL PLANE
#
#
#
###Example 1: Ref worst than desired in all metrics except the volker_metric
#dx = 0
#dy = 0
#x = np.linspace(450+dx,810+dx-0,2)
#y = np.linspace(300+dy,660+dy-0,2)
#
#x = np.linspace(500+dx,700+dx-0,2)
#y = np.linspace(400+dy,600+dy-0,2)
#xx,yy = np.meshgrid(x,y)
#square_imageCoordinates = np.array([xx.ravel(),yy.ravel(), np.ones_like(yy.ravel())])

#x = np.linspace(10, cam.img_width-10, 10)
#y = np.linspace(10, cam.img_height-10, 10)
#xx,yy = np.meshgrid(x,y)
#validation_imagePoints = np.array([xx.ravel(),yy.ravel(), np.ones_like(yy.ravel())])



""" This was inside the loop to test a set of square points in image coordinates"""
#    H_caminv = np.linalg.inv(H_cam)
#    square_ObjectPoints = np.dot(H_caminv,square_imageCoordinates)
#    square_ObjectPoints = np.insert(square_ObjectPoints,2,np.zeros(square_ObjectPoints.shape[1]), axis = 0)
#
#    
#for j in range(square_ObjectPoints.shape[1]):
#      square_ObjectPoints[0,j] = square_ObjectPoints[0,j]/square_ObjectPoints[3,j]
#      square_ObjectPoints[1,j] = square_ObjectPoints[1,j]/square_ObjectPoints[3,j]
#      square_ObjectPoints[2,j] = 0.
#      square_ObjectPoints[3,j] = 1.
##    H_caminv = np.linalg.inv(H_cam)
##
##    validation_objectPoints = np.dot(H_caminv, validation_imagePoints)
##    validation_objectPoints = np.insert(validation_objectPoints,2,np.zeros(validation_objectPoints.shape[1]), axis = 0)
##
##
##    ##NORMALIZE COORDINATES AND FIX THE Z TO ZERO (ALL POINTS ON A PLANE ON 3D COORDINATES)
##    for j in range(validation_objectPoints.shape[1]):
##      validation_objectPoints[0,j] = validation_objectPoints[0,j]/validation_objectPoints[3,j]
##      validation_objectPoints[1,j] = validation_objectPoints[1,j]/validation_objectPoints[3,j]
##      validation_objectPoints[2,j] = 0.
##      validation_objectPoints[3,j] = 1.