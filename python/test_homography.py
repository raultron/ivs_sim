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

      xx, yy, zz = uniform_sphere((0,360), (0,80), d, 4,4, False)

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


def h_norm2d(x):
  #Normalize points
  for i in range(3):
    x[i] = x[i]/x[2]
  return x

def d(x1, x2):
  return np.linalg.norm(h_norm2d(x1)-h_norm2d(x2))

def sym_transfer_error(Xo,Xi,H):
  """Symetric transfer error
  Xo: Object points in 2D Homogeneous Coordinates (3xn)
  Xi: Image points in 2D Homogeneous Coordinates (3xn)
  """
  Xo = np.copy(Xo)
  Xi = np.copy(Xi)
  H = np.copy(H)
  error1 = d(Xi,np.dot(H,Xo))
  error2 = d(Xo,np.dot(np.linalg.inv(H),Xi))
  return error1 + error2

def transfer_error(Xo,Xi,H):
  """transfer error including normalization
  Xo: Object points in 2D Homogeneous Coordinates (3xn)
  Xi: Image points in 2D Homogeneous Coordinates (3xn)
  """
  Xo = np.copy(Xo)
  Xi = np.copy(Xi)
  H = np.copy(H)
  return d(Xi,np.dot(H,Xo))

def algebraic_distance(Xo,Xi,H):
  """
  Xi point measured in the image
  Xo real value of the model point
  H an estimated homography
  as defined in Multiple View Geometry in Computer vision
  """
  Xo = np.copy(Xo)
  Xi = np.copy(Xi)
  H = np.copy(H)
#  a = np.cross(Xi,np.dot(H,Xo))
#  return a[0]**2 + a[1]**2
  Xio = np.dot(H,Xo)
  return (Xio[0]*Xi[2]-Xi[0]*Xio[2])**2 + (Xi[1]*Xio[2] - Xi[2]*Xio[1])**2

def geometric_distance(Xo,Xi,H):
  """
  Xi point measured in the image
  Xo real value of the model point
  H an estimated homography
  as defined in Multiple View Geometry in Computer vision
  """
  Xo = np.copy(Xo)
  Xi = np.copy(Xi)
  H = np.copy(H)
  Xio = np.dot(H,Xo)
  return np.sqrt((Xi[0]/Xi[2] - Xio[0]/Xio[2])**2+(Xi[1]/Xi[2] - Xio[1]/Xio[2])**2)


def calculate_A_matrix(Xo, Xi):
  """ Calculate the A matrix for the DLT algorithm:  A.H = 0
  Inputs:
    Xo: Object points in 3D Homogeneous Coordinates (3xn), Z coorinate removed
    since the points should be on a plane

    Xi: Image points in 2D Homogeneous Coordinates (3xn)
  """
  Npts = Xo.shape[1]
  A = np.zeros((2*Npts,9))
  O = np.zeros(3)

  for i in range(0, Npts):
      X = Xo[:,i].T
      u = Xi[0,i]
      v = Xi[1,i]
      w = Xi[2,i]
      A[2*i,:] = np.array([O, -w*X, v*X]).reshape(1, 9)
      A[2*i+1,:] = np.array([w*X, O, -u*X]).reshape(1, 9)
  return A
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
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(160.0))
cam_world = np.array([0.0,-0.2,0.4,1]).T
cam_t = np.dot(cam.R,-cam_world)
cam.set_t(cam_t[0], cam_t[1],  cam_t[2])
cam.set_P()


#Create a plane with 4 points to start
pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.2,0.2), n = (2,2))
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
y = np.linspace(400+dy,600+dy-0,2)
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
validation_plane =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.4,0.4), n = (4,4))
validation_plane.uniform()
x = np.linspace(10, cam.img_width-10, 10)
y = np.linspace(10, cam.img_height-10, 10)
xx,yy = np.meshgrid(x,y)
validation_imagePoints = np.array([xx.ravel(),yy.ravel(), np.ones_like(yy.ravel())])





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

    H_cam = cam.homography_from_Rt()
    H_caminv = np.linalg.inv(H_cam)
    square_ObjectPoints = np.dot(H_caminv,square_imageCoordinates)
    square_ObjectPoints = np.insert(square_ObjectPoints,2,np.zeros(square_ObjectPoints.shape[1]), axis = 0)

    for j in range(square_ObjectPoints.shape[1]):
      square_ObjectPoints[0,j] = square_ObjectPoints[0,j]/square_ObjectPoints[3,j]
      square_ObjectPoints[1,j] = square_ObjectPoints[1,j]/square_ObjectPoints[3,j]
      square_ObjectPoints[2,j] = 0.
      square_ObjectPoints[3,j] = 1.
#    H_caminv = np.linalg.inv(H_cam)
#
#    validation_objectPoints = np.dot(H_caminv, validation_imagePoints)
#    validation_objectPoints = np.insert(validation_objectPoints,2,np.zeros(validation_objectPoints.shape[1]), axis = 0)
#
#
#    ##NORMALIZE COORDINATES AND FIX THE Z TO ZERO (ALL POINTS ON A PLANE ON 3D COORDINATES)
#    for j in range(validation_objectPoints.shape[1]):
#      validation_objectPoints[0,j] = validation_objectPoints[0,j]/validation_objectPoints[3,j]
#      validation_objectPoints[1,j] = validation_objectPoints[1,j]/validation_objectPoints[3,j]
#      validation_objectPoints[2,j] = 0.
#      validation_objectPoints[3,j] = 1.

    validation_objectPoints =validation_plane.get_points()
    validation_imagePoints = np.array(cam.project(validation_objectPoints, False))

    #pl.uniform_with_distortion(mean = 0, sd = 0.02)
    pl.random(n = 16, r = 0.01, min_sep = 0.01)
    objectPoints_des = pl.get_points()
    objectPoints_des = square_ObjectPoints
    ## PROJECT OBJECT POINTS INTO CAMERA IMAGE
    imagePoints_ref = np.array(cam.project(objectPoints_ref, False))
    imagePoints_des = np.array(cam.project(objectPoints_des, False))



    ## ADD NOISE
    imagePoints_ref_noisy = imagePoints_ref
    imagePoints_ref_noisy = cam.addnoise_imagePoints(imagePoints_ref, mean = 0, sd = 2)

    imagePoints_des_noisy = imagePoints_des
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

    ## POINTS WITH A RANDOM CONFIGURATION FIRST

    ## FIX POINT FORMATS FOR USING THE FUNCTIONS
    Xo = np.copy(objectPoints_ref)
    Xo = np.delete(Xo, 2, axis=0)
    Xi = np.copy(imagePoints_ref_noisy)

    Aref = calculate_A_matrix(Xo, Xi)

    Href,A_t_ref,H_t = homo2d.homography2d(Xo,Xi)
    Href = Href/Href[2,2]

    ## CALCULATE HOMOGRAPHY ESTIMATION ERRORS
    ## we need points without noise to confirm
    ## object coordinates dont have noise, only image coordinates
    #validation_objectPoints = np.copy(objectPoints_ref)
    #validation_imagePoints =  np.copy(imagePoints_ref)

    Xo = np.copy(validation_objectPoints)
    Xo = np.delete(Xo, 2, axis=0)
    Xi = np.copy(validation_imagePoints)

    sum = 0
    for j in range(Xo.shape[1]):
        sum += geometric_distance(Xo[:,j],Xi[:,j],Href)
    transfer_error_ref.append(sum/Xo.shape[1])

    H_error = np.sqrt(np.sum(np.abs(H_cam - Href)**2))
    error_ref.append(H_error)

    ## NOW FOR POINTS WITH A DESIRED CONFIGURATION
    #Calculate the homography


    Xo = np.copy(objectPoints_des)
    Xo = np.delete(Xo, 2, axis=0)
    Xi = np.copy(imagePoints_des_noisy)

    Ades = calculate_A_matrix(Xo, Xi)

    Hdes,A_t_des,_ = homo2d.homography2d(Xo,Xi)
    Hdes = Hdes/Hdes[2,2]


    ## CALCULATE HOMOGRAPHY ESTIMATION ERRORS
    ## we need points without noise to confirm
    ## object coordinates dont have noise, only image coordinates
    #validation_objectPoints = np.copy(objectPoints_des)
    #validation_imagePoints =  np.copy(objectPoints_des)

    Xo = np.copy(validation_objectPoints)
    Xo = np.delete(Xo, 2, axis=0)
    Xi = np.copy(validation_imagePoints)

    sum = 0
    for j in range(Xo.shape[1]):
        sum += geometric_distance(Xo[:,j],Xi[:,j],Hdes)
    transfer_error_des.append(sum/Xo.shape[1])

    H_error = np.sqrt(np.sum(np.abs(H_cam - Hdes)**2))
    error_desired.append(H_error)


    Bref = np.insert(Aref,8,np.zeros(9),axis=0)
    Bs_ref = np.dot(Bref,Bref.T)

    Bdes = np.insert(Ades,8,np.zeros(9),axis=0)
    Bs_des = np.dot(Bdes,Bdes.T)


    volker_metric_ref.append(np.sqrt(np.sum( Bs_ref[[0,2,4,6],[1,3,5,7]]**2)))
    volker_metric_des.append(np.sqrt(np.sum( Bs_des[[0,2,4,6],[1,3,5,7]]**2)))


    ## This metric is based on the condition number (2-norm)
    ## Explained in chapter 5 section 2.2: http://www.math.ucla.edu/~dakuang/cse6040/lectures/6040_lecture15.pdf

    #More information about matrix perturbation theory
    #https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-241j-dynamic-systems-and-control-spring-2011/readings/MIT6_241JS11_chap05.pdf
    matrix_cond_ref.append(np.linalg.norm(A_t_ref,2)*np.linalg.norm(np.linalg.pinv(A_t_ref),2))
    matrix_cond_des.append(np.linalg.norm(A_t_des,2)*np.linalg.norm(np.linalg.pinv(A_t_des),2))

    algebraic_error_ref.append(np.linalg.norm(np.dot(Aref,H_cam.ravel().T)))
    algebraic_error_des.append(np.linalg.norm(np.dot(Ades,H_cam.ravel().T)))



transfer_error_des = np.array(transfer_error_des)
error_ref = np.array(error_ref)
error_desired = np.array(error_desired)

matrix_cond_ref = np.array(matrix_cond_ref)
matrix_cond_des = np.array(matrix_cond_des)

algebraic_error_ref = np.array(algebraic_error_ref)
algebraic_error_des = np.array(algebraic_error_des)

volker_metric_ref = np.array(volker_metric_ref)
volker_metric_des = np.array(volker_metric_des)



comp_abs_error = np.mean(error_ref) < np.mean(error_desired)
comp_transfer_error = np.mean(transfer_error_ref) < np.mean(transfer_error_des)
comp_matrix_conditioning = np.mean(matrix_cond_ref) < np.mean(matrix_cond_des)
comp_algebraic_error_mat = np.mean(algebraic_error_ref) < np.mean(algebraic_error_des)
comp_volker_metric = np.mean(volker_metric_ref) < np.mean(volker_metric_des)


comparison = np.array([comp_abs_error,comp_transfer_error,comp_matrix_conditioning,comp_algebraic_error_mat,comp_volker_metric])
print comparison

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


#if np.sum(comparison)>1:
#  print "Reference model (Black) is better in metrics:", comparison
#else:
#  print "Desired model (Red) is better in metrics:", ~comparison

#
##%%
#from sympy import *
#init_printing(use_unicode=True)
#
##Projection matrix (in symbolic py)
#P = Matrix(cam.P)
#
#
##fx = Symbol('fx')
##fy = Symbol('fy')
##cx = Symbol('cx')
##cy = Symbol('cy')
##
##K = Matrix([[fx, 0, cx],
##            [0,fy,cy],
##            [0,0,1]],)
##
##
##tx = Symbol('tx')
##ty = Symbol('ty')
##tz = Symbol('tz')
##
##r00 = Symbol('r00')
##r01 = Symbol('r01')
##r02 = Symbol('r02')
##
##r10 = Symbol('r10')
##r11 = Symbol('r11')
##r12 = Symbol('r12')
##
##r20 = Symbol('r20')
##r21 = Symbol('r21')
##r22 = Symbol('r22')
##
##Rt = Matrix([[r00, r01, r02, tx],
##             [r10, r11, r12, ty],
##            [r20, r21, r22, tz]])
##
##P = K*Rt
#
##points in model plane
#
#x1 = Symbol('x1')
#y1 = Symbol('y1')
#l1 = Symbol('l1')
#ex1 = Symbol('ex1')
#X = Matrix([x1,y1,0,l1])
#X = Matrix([x1,y1,0,1])
#U = P*X
#u1 = U[0]/U[2]
#v1 = U[1]/U[2]
#w1 = U[2]/U[2]
#
#x2 = Symbol('x2')
#y2 = Symbol('y2')
#l2 = Symbol('l2')
#ex2 = Symbol('ex2')
#X = Matrix([x2,y2,0,l2])
#X = Matrix([x2,y2,0,1])
#U = P*X
#u2 = U[0]/U[2]
#v2 = U[1]/U[2]
#w2 = U[2]/U[2]
#
#x3 = Symbol('x3')
#y3 = Symbol('y3')
#l3 = Symbol('l3')
#ex3 = Symbol('ex3')
#X = Matrix([x3,y3,0,l3])
#X = Matrix([x3,y3,0,1])
#U = P*X
#u3 = U[0]/U[2]
#v3 = U[1]/U[2]
#w3 = U[2]/U[2]
#
#x4 = Symbol('x4')
#y4 = Symbol('y4')
#l4 = Symbol('l4')
#ex4 = Symbol('ex4')
#X = Matrix([x4,y4,0,l4])
#X = Matrix([x4,y4,0,1])
#U = P*X
#u4 = U[0]/U[2]
#v4 = U[1]/U[2]
#w4 = U[2]/U[2]
#
##      X = x1[:,i].T
##      x = x2[0,i]
##      y = x2[1,i]
##      w = x2[2,i]
##      Ades[2*i,:] = np.array([O, -w*X, y*X]).reshape(1, 9)
##      Ades[2*i+1,:] = np.array([w*X, O, -x*X]).reshape(1, 9)
#
##Asymb = Matrix([[   0,    0,     0, -w1*x1, -w1*y1, -w1*l1,  v1*x1,  v1*y1,  v1*l1],
##                [w1*x1, w1*y1, w1*l1,      0,      0,      0, -u1*x1, -u1*y1, -u1*l1],
##
##                [   0,    0,     0, -w2*x2, -w2*y2, -w2*l2,  v2*x2,  v2*y2,  v2*l2],
##                [w2*x2, w2*y2, w2*l2,      0,      0,      0, -u2*x2, -u2*y2, -u2*l2],
##
##                [   0,    0,     0, -w3*x3, -w3*y3, -w3*l3,  v3*x3,  v3*y3,  v3*l3],
##                [w3*x3, w3*y3, w3*l3,      0,      0,      0, -u3*x3, -u3*y3, -u3*l3],
##
##                [   0,    0,     0, -w4*x4, -w4*y4, -w4*l4,  v4*x4,  v4*y4,  v4*l4],
##                [w4*x4, w4*y4, w4*l4,      0,      0,      0, -u4*x4, -u4*y4, -u4*l4],
##
##                [0, 0, 0,      0,      0,      0, 0, 0, 0],
##        ])
#
#
## If we assume that object and image coordinates are normalized
#Asymb = Matrix([[   0,    0,     0, -x1, -y1, -1,  v1*x1,  v1*y1,  v1],
#                [x1, y1, 1,      0,      0,      0, -u1*x1, -u1*y1, -u1],
#
#                [   0,    0,     0, -x2, -y2, -1,  v2*x2,  v2*y2,  v2],
#                [x2, y2, 1,      0,      0,      0, -u2*x2, -u2*y2, -u2],
#
#                [   0,    0,     0, -x3, -y3, -1,  v3*x3,  v3*y3,  v3],
#                [x3, y3, 1,      0,      0,      0, -u3*x3, -u3*y3, -u3],
#
#                [   0,    0,     0, -x4, -y4, -1,  v4*x4,  v4*y4,  v4],
#                [x4, y4, 1,      0,      0,      0, -u4*x4, -u4*y4, -u4],
#
#
#        ])
#
#               # [0, 0, 0,      0,      0,      0, 0, 0, 0],
#
##%%
#do = objectPoints_des
#A_test = np.array(Asymb.evalf(subs={x1: do[0,0], y1: do[1,0], l1: do[3,0],
#                           x2: do[0,1], y2: do[1,1], l2: do[3,1],
#                           x3: do[0,2], y3: do[1,2], l3: do[3,2],
#                           x4: do[0,3], y4: do[1,3], l4: do[3,3]})).astype(np.float64)
#
##
##Bs_sym = Matrix(Asymb.T.dot(Asymb)).reshape(9,9)
##Bs_sym_test = np.array(Bs_sym.evalf(subs={x1: do[0,0], y1: do[1,0], l1: do[3,0],
##                           x2: do[0,1], y2: do[1,1], l2: do[3,1],
##                           x3: do[0,2], y3: do[1,2], l3: do[3,2],
##                           x4: do[0,3], y4: do[1,3], l4: do[3,3]})).astype(np.float64)
##
###points in image coordinates
##print np.allclose(Ades, A_test[:8,:])
##print np.allclose(Bs2, Bs_sym_test)
#
##
##volker_metric_ref = 0
##start = 1
##stop =8
##for i in range(7):
##  for j in range(start,stop):
##    volker_metric_ref = volker_metric_ref + sqrt(Bs_sym[i,j]**2)
##  start = start +1
##Asymb_pinv = Asymb.H * (Asymb * Asymb.H) ** -1
#
#Asymb_vec = Asymb.vec()
#vals = list(Asymb_vec.values()) or [0]
#matrix_cond_sym = sqrt(Add(*(i**2 for i in vals)))
#
#algebraic_error = Asymb*H_cam.ravel().T
#vals = list(algebraic_error.values()) or [0]
#algebraic_error_symb = sqrt(Add(*(i**2 for i in vals)))
#
##matrix_cond_sym = Asymb.norm()**2#norm((Asymb_pinv),2)
##%%
#op = np.copy(objectPoints_ref)
#
#print matrix_cond_sym.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#
#d_x1 = diff(matrix_cond_sym,x1)
#d_y1 = diff(matrix_cond_sym,y1)
#d_x2 = diff(matrix_cond_sym,x2)
#d_y2 = diff(matrix_cond_sym,y2)
#d_x3 = diff(matrix_cond_sym,x3)
#d_y3 = diff(matrix_cond_sym,y3)
#d_x4 = diff(matrix_cond_sym,x4)
#d_y4 = diff(matrix_cond_sym,y4)
#
##print d_x1.evalf(subs={x1: do[0,0], y1: do[1,0],
##                           x2: do[0,1], y2: do[1,1],
##                           x3: do[0,2], y3: do[1,2],
##                           x4: do[0,3], y4: do[1,3]})
#
#op = np.copy(objectPoints_ref)
#d_x1_eval = d_x1.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#
#d_y1_eval = d_y1.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#
#d_x2_eval = d_x2.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#d_y2_eval = d_y2.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#d_x3_eval = d_x3.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#d_y3_eval = d_y3.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#d_x4_eval = d_x4.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#d_y4_eval = d_y4.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#
#
#
#
#op = np.copy(objectPoints_ref)
#
#alpha = 1E-4
#op[0,0] = op[0,0] - d_x1_eval*alpha
#op[1,0] = op[1,0] - d_y1_eval*alpha
#
#op[0,1] = op[0,1] - d_x2_eval*alpha
#op[1,1] = op[1,1] - d_y2_eval*alpha
#
#op[0,2] = op[0,2] - d_x3_eval*alpha
#op[1,2] = op[1,2] - d_y3_eval*alpha
#
#op[0,3] = op[0,3] - d_x4_eval*alpha
#op[1,2] = op[1,3] - d_y4_eval*alpha
#
#print (d_x1_eval*alpha, d_y1_eval*alpha)
#print (d_x2_eval*alpha, d_y2_eval*alpha)
#print (d_x3_eval*alpha, d_y3_eval*alpha)
#print (d_x4_eval*alpha, d_y4_eval*alpha)
#
#
#print matrix_cond_sym.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#
#imagePoints2 = np.array(cam.project(op, False))
#
#plt.plot(imagePoints2[0],imagePoints2[1],'.',color = 'g',)
#plt.plot(imagePoints_ref[0],imagePoints_ref[1],'.',color = 'black',)
#plt.pause(0.05)
