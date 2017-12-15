# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:30:20 2017

@author: racuna
"""


### TODO LIST

### Inlcude different cameras in the simulation,
## I think they will make a difference.

from pose_sim import *
from vision.camera import *
from vision.plane import Plane
from vision.screen import Screen
from ippe import homo2d
from error_functions import geometric_distance_points, get_matrix_conditioning_number, volker_metric

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from Rt_matrix_from_euler_t import R_matrix_from_euler_t
from uniform_sphere import uniform_sphere

def create_cam_distribution(cam = None, plane = None, deviation = 0, plot=False):
  if cam == None:
    # Create an initial camera on the center of the world
    cam = Camera()
    f = 800
    cam.set_K(fx = f, fy = f, cx = 320, cy = 240)  #Camera Matrix
    cam.img_width = 320*2
    cam.img_height = 240*2

  if plane == None:
    # we create a default plane with 4 points with a side lenght of w (meters)
    w = 0.17
    plane =  Plane(origin=np.array([0, 0, 0] ), normal = np.array([0, 0, 1]), size=(w,w), n = (2,2))
  else:
    if deviation > 0:
      #We extend the size of this plane to account for the deviation from a uniform pattern
      plane.size = (plane.size[0] + deviation, plane.size[1] + deviation)


  # We create an uniform distribution of points in image coordinates
  x_min = 0
  x_max = cam.img_width
  y_min = 0
  y_max = cam.img_height
  x_dist = np.linspace(x_min,x_max, 3)
  y_dist = np.linspace(y_min,y_max,3)
  xx, yy = np.meshgrid(x_dist, y_dist)
  hh = np.ones_like(xx, dtype=np.float32)
  imagePoints = np.array([xx.ravel(),yy.ravel(), hh.ravel()], dtype=np.float32)

  # Backproject the pixels into rays (unit vector with the tail at the camera center)
  Kinv = np.linalg.inv(cam.K)
  unit_rays = np.array(np.dot(Kinv,imagePoints))

  #fig = plt.figure()
  #ax = fig.add_subplot(111, projection='3d')

  #origin = np.zeros_like(unit_rays[0,:])
  #ax.quiver(origin,origin,origin,unit_rays[0,:],unit_rays[1,:],unit_rays[2,:], length=1.0, pivot = 'tail')
  #ax.scatter(unit_rays[0,:],unit_rays[1,:],unit_rays[2,:], 'rx')

  #Select a linear space of distances based on focal length (like in IPPE paper)
  #d_space = np.linspace(f/2,2*f, 4)

  #Select a linear space of distances based on focal length (like in IPPE paper)
  d_space = np.linspace(0.25,1.0,4)

  #t = d*unit_rays;
  t_list = []
  for d in d_space:

      #t_list.append(d*unit_rays)

      xx, yy, zz = uniform_sphere((0,360,4), (0,80,4), d, False) # uniform_sphere((0,360), (0,80), d, 4,4, False)

      sphere_points = np.array([xx.ravel(),yy.ravel(), zz.ravel()], dtype=np.float32)

      t_list.append(sphere_points)

  t_space = np.hstack(t_list)

  #we now create a plane model for each t
  pl_space= []
  for t in t_space.T:
    pl = plane.clone()
    pl.set_origin(np.array([t[0], t[1], t[2]]))
    pl.uniform()
    pl_space.append(pl)

  #ax.scatter(t_space[0,:],t_space[1,:],t_space[2,:], color = 'b')

  for pl in pl_space:
    objectPoints = pl.get_points()
    #ax.scatter(objectPoints[0,:],objectPoints[1,:],objectPoints[2,:], color = 'r')

  cams = []
  for pl in pl_space:

    cam = cam.clone()
    cam.set_t(-pl.origin[0], -pl.origin[1],-pl.origin[2])
    cam.set_R_mat(R_matrix_from_euler_t(0.0,0,0))
    cam.look_at([0,0,0])

    #

    pl.set_origin(np.array([0, 0, 0]))
    pl.uniform()
    objectPoints = pl.get_points()
    imagePoints = cam.project(objectPoints)
    if plot:
      cam.plot_image(imagePoints)
    if ((imagePoints[0,:]<cam.img_width) & (imagePoints[0,:]>0)).all():
      if ((imagePoints[1,:]<cam.img_height) & (imagePoints[1,:]>0)).all():
        cams.append(cam)

  if plot:
    planes = []
    pl.uniform()
    planes.append(pl)
    plot3D(cams, planes)

  return cams





#############################################
## INITIALIZATIONS
#############################################

## CREATE A SIMULATED CAMERA
cam = Camera()
fx = fy =  800
cx = 640
cy = 480
cam.set_K(fx,fy,cx,cy)
cam.img_width = 1280
cam.img_height = 960

## DEFINE CAMERA POSE LOOKING STRAIGTH DOWN INTO THE PLANE MODEL
cam.set_R_axisAngle(1.0,  1.0,  0.0, np.deg2rad(165.0))
cam_world = np.array([0.0,-0.2,1,1]).T
cam_t = np.dot(cam.R,-cam_world)
cam.set_t(cam_t[0], cam_t[1],  cam_t[2])
cam.set_P()
H_cam = cam.homography_from_Rt()


#Create a plane with 4 points to start
pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.3,0.3), n = (2,2))
pl.uniform()

cams = create_cam_distribution(cam, pl, 0.01, plot=False)
#pl.uniform_with_distortion(mean = 0, sd = 0.01)



## OBTAIN PLANE MODEL POINTS
objectPoints_ref = pl.get_points()



#objectPoints_ref = np.load('objectPoints_test_ippe5.npy')

## CREATE AN IDEAL SET OF POINTS IN IMAGE COORDINATES (A SQUARE)
## FROM THE DESIRED IMAGE POINTS AND THE CAMERA MATRIX OBTAIN THE COORDINATES IN
## MODEL PLANE



##Example 1: Ref worst than desired in all metrics except the volker_metric
dx = 0
dy = 0
x = np.linspace(450+dx,810+dx-0,2)
y = np.linspace(300+dy,660+dy-0,2)

x = np.linspace(500+dx,700+dx-0,2)
y = np.linspace(400+dy,6+dy-0,2)
xx,yy = np.meshgrid(x,y)
square_imageCoordinates = np.array([xx.ravel(),yy.ravel(), np.ones_like(yy.ravel())])
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


## CREATE A SET OF IMAGE POINTS FOR VALIDATION OF THE HOMOGRAPHY ESTIMATION
validation_plane =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.4,0.4), n = (8,8))
validation_plane.uniform()
x = np.linspace(10, cam.img_width-10, 10)
y = np.linspace(10, cam.img_height-10, 10)
xx,yy = np.meshgrid(x,y)
validation_imagePoints = np.array([xx.ravel(),yy.ravel(), np.ones_like(yy.ravel())])
validation_objectPoints =validation_plane.get_points()
validation_imagePoints = np.array(cam.project(validation_objectPoints, False))




#############################################
## WE START THE MAIN LOOP OF CALCULATIONS
#############################################

H_error_best_configurations = list()

geometric_distance_old = 1000
max_iters = 100
geometric_distance_iter = 0
H_error = 0
last_best = "uniform"
current = 0
best_configurations = list()

for j in range(max_iters):
  if last_best == "uniform":
    pl.random(n =4, r = 0.01, min_sep = 0.01)
    current = "random"

  else:
    pl.uniform()
    #pl.random(n = 4, r = 0.01, min_sep = 0.01)
    current = "uniform"

    #pl.uniform_with_distortion(mean = 0, sd = 0.02)


  objectPoints_des = pl.get_points()
  imagePoints_des = np.array(cam.project(objectPoints_des, False))
  #if j == 0:
  #   imagePoints_des = best_imagePoints
   #  objectPoints_des = best_objectPoints
  geometric_distance_list = list()
  H_error_list = list()
  matrix_cond_des = list()
  algebraic_error_des = list()
  volker_metric_list = list()
  for i in range(100):

    imagePoints_des_noisy = cam.addnoise_imagePoints(imagePoints_des, mean = 0, sd = 4)

    ## NOW FOR EACH SET OF POINTS WE CALCULATE THE HOMOGRAPHY USING THE DLT ALGORITHM

      ## NOW FOR POINTS WITH A DESIRED CONFIGURATION
    #Calculate the homography


    Xo = np.copy(objectPoints_des[[0,1,3],:]) #without the z coordinate (plane)
    Xi = np.copy(imagePoints_des_noisy)
    Ades = calculate_A_matrix(Xo, Xi)

    #Ades = calculate_A_matrix(Xo, Xi)

    Hdes,A_t_des,_ = homo2d.homography2d(Xo,Xi)
    Hdes = Hdes/Hdes[2,2]


    ## CALCULATE HOMOGRAPHY ESTIMATION ERRORS
    ## we need points without noise to confirm
    ## object coordinates dont have noise, only image coordinates
    #validation_objectPoints = np.copy(objectPoints_des)
    #validation_imagePoints =  np.copy(objectPoints_des)

    Xo = np.copy(validation_objectPoints[[0,1,3],:]) #without the z coordinate (plane)
    Xi = np.copy(validation_imagePoints)

    geometric_distance_list.append(geometric_distance_points(Xo,Xi,Hdes))
    H_error_list.append(np.sqrt(np.sum(np.abs(H_cam - Hdes)**2)))


    Xo = np.copy(objectPoints_des[[0,1,3],:]) #without the z coordinate (plane)
    Xi = np.copy(imagePoints_des)
    Aideal = calculate_A_matrix(Xo, Xi)

    As_ideal, volkerMetric = volker_metric(Aideal)
    volker_metric_list.append(volkerMetric)

    ## This metric is based on the condition number (2-norm)
    ## Explained in chapter 5 section 2.2: http://www.math.ucla.edu/~dakuang/cse6040/lectures/6040_lecture15.pdf

    matrix_cond_des.append(get_matrix_conditioning_number(A_t_des))


    algebraic_error_des.append(np.linalg.norm(np.dot(Ades,H_cam.ravel().T)))


  geometric_distance_iter = np.mean(geometric_distance_list)
  if  geometric_distance_iter < geometric_distance_old:
    print current
    last_best = current
    best_imagePoints = imagePoints_des
    best_objectPoints = objectPoints_des
    geometric_distance_old = geometric_distance_iter
    best_configurations.append(imagePoints_des)
    plt.ion()
    plt.cla()
    plt.figure('Image Points')
    cam.plot_plane(pl)
    plt.plot(imagePoints_des_noisy[0],imagePoints_des_noisy[1],'x',color = 'r',  label='Desired',)
    plt.xlim(0,1280)
    plt.ylim(0,960)
      #if i==0:
      #    plt.legend()
    plt.gca().invert_yaxis()
    plt.pause(0.01)
    H_error_best_configurations.append(H_error)




    print "iter: ", j
    print "--------------------------------------------------------"
    print "H Matrix absolute Error"
    print "Des: ", np.mean(H_error_list)
    print "--------------------------------------------------------"

    print "Geometric distance for the validation points"
    print "Des: ", geometric_distance_iter
    print "--------------------------------------------------------"

    print "Matrix conditioning"
    print "Des: ", np.mean(matrix_cond_des)
    print "--------------------------------------------------------"

    print "Volker Matrix"
    print "Des: ", np.mean(volker_metric_list)
    print "--------------------------------------------------------"

  geometric_distance_iter = 0
  H_error = 0






