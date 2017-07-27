# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:30:20 2017

@author: racuna
"""


### TODO LIST

### 1. Confirm that the error calculation is OK
### 2. Test the symmetrical transfer error with the whole space of image coordinates
### using simulated true data (not only with the 4 points)
### 3. Include the transformation of coordinates on the symbolic math


from vision.camera import *
from vision.plane import Plane
from vision.screen import Screen
from ippe import homo2d

import numpy as np
import matplotlib.pyplot as plt


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
cam.set_R_axisAngle(0.0,  1.0,  0.0, np.deg2rad(170.0))
cam_world = np.array([0.0,0.0,0.5,1]).T
cam_t = np.dot(cam.R,-cam_world)
cam.set_t(cam_t[0], cam_t[1],  cam_t[2])
cam.set_P()

## TRUE VALUE OF HOMOGRAPHY OBTAINED FROM CAMERA PARAMETERS
H_cam = cam.homography_from_Rt()


#Create a plane with 4 points to start
#pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.2,0.2), n = (2,2))
#pl.uniform()
#pl.uniform_with_distortion(mean = 0, sd = 0.01)


## OBTAIN PLANE MODEL POINTS
#objectPoints = pl.get_points()
objectPoints = np.load('objectPoints_test_ippe.npy')

## CREATE AN IDEAL SET OF POINTS IN IMAGE COORDINATES (A SQUARE)
## FROM THE DESIRED IMAGE POINTS AND THE CAMERA MATRIX OBTAIN THE COORDINATES IN
## MODEL PLANE



#Example 1: Ref worst than desired in all metrics except the volker_metric
dx = 0
dy = 0
x = np.linspace(185+dx,805+dx-0,2)
y = np.linspace(235+dy,752+dy-0,2)

#Example 2: Ref better than desired in all metrics except the volker_metric
dx = 0
dy = 0
x = np.linspace(185+dx,805+dx-400,2)
y = np.linspace(235+dy,752+dy-300,2)

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

xx,yy = np.meshgrid(x,y)
desired_imageCoordinates = np.array([xx.ravel(),yy.ravel(), np.ones_like(yy.ravel())])

desired_objectPoints = np.dot(np.linalg.inv(H_cam),desired_imageCoordinates)
desired_objectPoints = np.insert(desired_objectPoints,2,np.zeros(desired_objectPoints.shape[1]), axis = 0)


##NORMALIZE COORDINATES AND FIX THE Z TO ZERO (ALL POINTS ON A PLANE ON 3D COORDINATES)
for i in range(desired_objectPoints.shape[1]):
  desired_objectPoints[0,i] = desired_objectPoints[0,i]/desired_objectPoints[3,i]
  desired_objectPoints[1,i] = desired_objectPoints[1,i]/desired_objectPoints[3,i]
  desired_objectPoints[2,i] = 0
  desired_objectPoints[3,i] = 1


## CREATE A SET OF IMAGE POINTS FOR VALIDATION OF THE HOMOGRAPHY ESTIMATION
x = np.linspace(10, cam.img_width-10, 10)
Y = np.linspace(10, cam.img_height-10, 10)
xx,yy = np.meshgrid(x,y)
validation_imagePoints = np.array([xx.ravel(),yy.ravel(), np.ones_like(yy.ravel())])
validation_objectPoints = np.dot(np.linalg.inv(H_cam),validation_imagePoints)
validation_objectPoints = np.insert(validation_objectPoints,2,np.zeros(validation_objectPoints.shape[1]), axis = 0)


##NORMALIZE COORDINATES AND FIX THE Z TO ZERO (ALL POINTS ON A PLANE ON 3D COORDINATES)
for i in range(validation_objectPoints.shape[1]):
  validation_objectPoints[0,i] = validation_objectPoints[0,i]/validation_objectPoints[3,i]
  validation_objectPoints[1,i] = validation_objectPoints[1,i]/validation_objectPoints[3,i]
  validation_objectPoints[2,i] = 0
  validation_objectPoints[3,i] = 1




#############################################
## WE START THE MAIN LOOP OF CALCULATIONS
#############################################

error_normal = list()
error_desired = list()
sym_transfer_error1 = list()
sym_transfer_error2 = list()
max_iters = 100
metric_base = 0
metric_desired = 0
matrix_cond_des = list()
matrix_cond_ref = list()

for i in range(max_iters):
  ## PROJECT OBJECT POINTS INTO CAMERA IMAGE
  imagePoints = np.array(cam.project(objectPoints, False))
  desired_imagePoints = np.array(cam.project(desired_objectPoints, False))

  ## ADD NOISE
  imagePoints_noisy = imagePoints
  imagePoints_noisy = cam.addnoise_imagePoints(imagePoints, mean = 0, sd = 4)

  desired_imagePoints_noisy = desired_imagePoints
  desired_imagePoints_noisy = cam.addnoise_imagePoints(desired_imagePoints, mean = 0, sd = 4)

  ## SHOW THE PROJECTIONS IN IMAGE PLANE
  plt.ion()
  plt.figure('Image Points')
  plt.plot(imagePoints_noisy[0],imagePoints_noisy[1],'.',color = 'r',)
  plt.plot(desired_imagePoints_noisy[0],desired_imagePoints_noisy[1],'x',color = 'b',)
  plt.xlim(0,1280)
  plt.ylim(0,960)
  plt.gca().invert_yaxis()
  #plt.pause(0.05)

  ## NOW FOR EACH SET OF POINTS WE CALCULATE THE HOMOGRAPHY USING THE DLT ALGORITHM

  ## POINTS WITH A RANDOM CONFIGURATION FIRST

  ## FIX POINT FORMATS FOR USING THE FUNCTIONS
  Xo = np.copy(objectPoints)
  Xo = np.delete(Xo, 2, axis=0)
  Xi = np.copy(imagePoints_noisy)

  A1 = calculate_A_matrix(Xo, Xi)

  H1,A_t1,H_t = homo2d.homography2d(Xo,Xi)
  H1 = H1/H1[2,2]

  ## CALCULATE HOMOGRAPHY ESTIMATION ERRORS
  ## we need points without noise to confirm
  ## object coordinates dont have noise, only image coordinates
  Xo = np.copy(validation_objectPoints)
  Xo = np.delete(Xo, 2, axis=0)
  Xi = np.copy(validation_imagePoints)

  sum = 0
  for i in range(Xo.shape[1]):
      sum += sym_transfer_error(Xo[:,i],Xi[:,i],H1)
  sym_transfer_error1.append(sum/Xo.shape[1])

  H_error = np.sum(np.abs(H_cam - H1))
  error_normal.append(H_error)

  ## NOW FOR POINTS WITH AN IDEAL CONFIGURATION (SQUARE)
  #Calculate the homography

  Xo = np.copy(desired_objectPoints)
  Xo = np.delete(Xo, 2, axis=0)
  Xi = np.copy(desired_imagePoints_noisy)

  A2 = calculate_A_matrix(Xo, Xi)

  H2,A_t2,_ = homo2d.homography2d(Xo,Xi)
  H2 = H2/H2[2,2]


  ## CALCULATE HOMOGRAPHY ESTIMATION ERRORS
  ## we need points without noise to confirm
  ## object coordinates dont have noise, only image coordinates
  Xo = np.copy(validation_objectPoints)
  Xo = np.delete(Xo, 2, axis=0)
  Xi = np.copy(validation_imagePoints)

  sum = 0
  for i in range(Xo.shape[1]):
      sum += sym_transfer_error(Xo[:,i],Xi[:,i],H2)
  sym_transfer_error2.append(sum/Xo.shape[1])

  H_error = np.sum(np.abs(H_cam - H2))
  error_desired.append(H_error)


  B1 = np.insert(A1,8,np.zeros(9),axis=0)
  Bs1 = np.dot(B1,B1.T)

  B2 = np.insert(A2,8,np.zeros(9),axis=0)
  Bs2 = np.dot(B2,B2.T)


  start = 1
  stop = 8
  for i in range(7):
    metric_base = metric_base + np.sum( np.abs(Bs1[i,start:stop]))
    metric_desired = metric_desired + np.sum(np.abs(Bs2[i,start:stop]))
    start = start +1
  metric_base -= np.sum( np.abs(Bs1[[0,2,4,6],[1,3,5,7]]))
  metric_desired -= np.sum( np.abs(Bs2[[0,2,4,6],[1,3,5,7]]))


  ## This metric is based on the condition number (2-norm)
  ## Explained in chapter 5 section 2.2: http://www.math.ucla.edu/~dakuang/cse6040/lectures/6040_lecture15.pdf
  matrix_cond_ref.append(np.linalg.norm(A1,2)*np.linalg.norm(np.linalg.pinv(A1),2))
  matrix_cond_des.append(np.linalg.norm(A2,2)*np.linalg.norm(np.linalg.pinv(A2),2))



sym_transfer_error2 = np.array(sym_transfer_error2)
error_normal = np.array(error_normal)
error_desired = np.array(error_desired)

matrix_cond_ref = np.array(matrix_cond_ref)
matrix_cond_des = np.array(matrix_cond_des)


comp = error_normal < error_desired
print "Error in homography estimation"
print "Times normal pattern was better than ideal pattern: ", np.count_nonzero(comp)
print "error normal : ", np.sum(error_normal)/len(error_normal)
print "error desired: ", (np.sum(error_desired)/len(error_desired))

print "Symmetric transfer error for the validation points"
print "sym transfer error ref: ", np.sum(sym_transfer_error1)/len(sym_transfer_error1)
print "sym transfer error des: ", np.sum(sym_transfer_error2)/len(sym_transfer_error2)

print "Matrix conditioning ref: ", np.mean(matrix_cond_ref)
print "Matrix conditioning des: ", np.mean(matrix_cond_des)





print 'metric  normal: ', metric_base/max_iters
print 'metric desired: ', metric_desired/max_iters


##%%
#from sympy import *
#init_printing(use_unicode=True)
#
##Projection matrix (in symbolic py)
#P = Matrix(cam.P)
#
##points in model plane
#x1 = Symbol('x1')
#y1 = Symbol('y1')
#l1 = Symbol('l1')
#X = Matrix([x1,y1,0,l1])
#X = Matrix([x1,y1,0,1])
#U = P.dot(X)
#u1 = U[0]/U[2]
#v1 = U[1]/U[2]
#w1 = U[2]/U[2]
#
#x2 = Symbol('x2')
#y2 = Symbol('y2')
#l2 = Symbol('l2')
#X = Matrix([x2,y2,0,l2])
#X = Matrix([x2,y2,0,1])
#U = P.dot(X)
#u2 = U[0]/U[2]
#v2 = U[1]/U[2]
#w2 = U[2]/U[2]
#
#x3 = Symbol('x3')
#y3 = Symbol('y3')
#l3 = Symbol('l3')
#X = Matrix([x3,y3,0,l3])
#X = Matrix([x3,y3,0,1])
#U = P.dot(X)
#u3 = U[0]/U[2]
#v3 = U[1]/U[2]
#w3 = U[2]/U[2]
#
#x4 = Symbol('x4')
#y4 = Symbol('y4')
#l4 = Symbol('l4')
#X = Matrix([x4,y4,0,l4])
#X = Matrix([x4,y4,0,1])
#U = P.dot(X)
#u4 = U[0]/U[2]
#v4 = U[1]/U[2]
#w4 = U[2]/U[2]
#
##      X = x1[:,i].T
##      x = x2[0,i]
##      y = x2[1,i]
##      w = x2[2,i]
##      A2[2*i,:] = np.array([O, -w*X, y*X]).reshape(1, 9)
##      A2[2*i+1,:] = np.array([w*X, O, -x*X]).reshape(1, 9)
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
#                [0, 0, 0,      0,      0,      0, 0, 0, 0],
#        ])
#
#
#
#do = desired_objectPoints
#A_test = np.array(Asymb.evalf(subs={x1: do[0,0], y1: do[1,0], l1: do[3,0],
#                           x2: do[0,1], y2: do[1,1], l2: do[3,1],
#                           x3: do[0,2], y3: do[1,2], l3: do[3,2],
#                           x4: do[0,3], y4: do[1,3], l4: do[3,3]})).astype(np.float64)
#
#
#Bs_sym = Matrix(Asymb.T.dot(Asymb)).reshape(9,9)
#Bs_sym_test = np.array(Bs_sym.evalf(subs={x1: do[0,0], y1: do[1,0], l1: do[3,0],
#                           x2: do[0,1], y2: do[1,1], l2: do[3,1],
#                           x3: do[0,2], y3: do[1,2], l3: do[3,2],
#                           x4: do[0,3], y4: do[1,3], l4: do[3,3]})).astype(np.float64)
#
##points in image coordinates
#print np.allclose(A2, A_test[:8,:])
#print np.allclose(Bs2, Bs_sym_test)
#
#
#metric_base = 0
#start = 1
#stop =8
#for i in range(7):
#  for j in range(start,stop):
#    metric_base = metric_base + sqrt(Bs_sym[i,j]**2)
#  start = start +1
#
#
#op = np.copy(objectPoints)
#
#print metric_base.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#
#d_x1 = diff(metric_base,x1)
#d_y1 = diff(metric_base,y1)
#d_x2 = diff(metric_base,x2)
#d_y2 = diff(metric_base,y2)
#d_x3 = diff(metric_base,x3)
#d_y3 = diff(metric_base,y3)
#d_x4 = diff(metric_base,x4)
#d_y4 = diff(metric_base,y4)
#
##print d_x1.evalf(subs={x1: do[0,0], y1: do[1,0],
##                           x2: do[0,1], y2: do[1,1],
##                           x3: do[0,2], y3: do[1,2],
##                           x4: do[0,3], y4: do[1,3]})
#
#op = np.copy(objectPoints)
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
#op = np.copy(objectPoints)
#
#alpha = 1E-8
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
#print metric_base.evalf(subs={x1: op[0,0], y1: op[1,0],
#                           x2: op[0,1], y2: op[1,1],
#                           x3: op[0,2], y3: op[1,2],
#                           x4: op[0,3], y4: op[1,3]})
#
#imagePoints2 = np.array(cam.project(op, False))
#
#plt.plot(imagePoints2[0],imagePoints2[1],'.',color = 'g',)
#plt.pause(0.05)
