#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:46:12 2017

@author: lracuna
"""
import numpy as np
from vision.camera import *
from vision.plane import Plane

from sympy import Matrix, Symbol,init_printing, sqrt,diff
init_printing(use_unicode=True)










def projection_matrix_symb():
  fx = Symbol('fx')
  fy = Symbol('fy')
  cx = Symbol('cx')
  cy = Symbol('cy')

  K = Matrix([[fx,   0, cx],
              [ 0,  fy, cy],
              [ 0,   0,  1]],)


  tx = Symbol('tx')
  ty = Symbol('ty')
  tz = Symbol('tz')

  r00 = Symbol('r00')
  r01 = Symbol('r01')
  r02 = Symbol('r02')

  r10 = Symbol('r10')
  r11 = Symbol('r11')
  r12 = Symbol('r12')

  r20 = Symbol('r20')
  r21 = Symbol('r21')
  r22 = Symbol('r22')

  Rt = Matrix([[r00, r01, r02, tx],
               [r10, r11, r12, ty],
               [r20, r21, r22, tz]])

  P = K*Rt
  return P


def evaluate_model_points(SymbMatrix, objectPoints):
  op = objectPoints
  x1 = Symbol('x1')
  y1 = Symbol('y1')
  l1 = Symbol('l1')


  x2 = Symbol('x2')
  y2 = Symbol('y2')
  l2 = Symbol('l2')


  x3 = Symbol('x3')
  y3 = Symbol('y3')
  l3 = Symbol('l3')


  x4 = Symbol('x4')
  y4 = Symbol('y4')
  l4 = Symbol('l4')

  SymbMatrix_eval = np.array(SymbMatrix.evalf(subs={x1: op[0,0], y1: op[1,0], l1: op[3,0],
                           x2: op[0,1], y2: op[1,1], l2: op[3,1],
                           x3: op[0,2], y3: op[1,2], l3: op[3,2],
                           x4: op[0,3], y4: op[1,3], l4: op[3,3]})).astype(np.float64)
  return SymbMatrix_eval


def create_A_symb(ProjectionMatrix):
  #Projection matrix (in symbolic py)
  P = Matrix(ProjectionMatrix)
  #create points in model plane (only 4 points configuration)
  x1 = Symbol('x1')
  y1 = Symbol('y1')
  l1 = Symbol('l1')
  #X1 = Matrix([x1,y1,0,l1])
  X1 = Matrix([x1,y1,0,1])

  x2 = Symbol('x2')
  y2 = Symbol('y2')
  l2 = Symbol('l2')
  #X = Matrix([x2,y2,0,l2])
  X2 = Matrix([x2,y2,0,1])

  x3 = Symbol('x3')
  y3 = Symbol('y3')
  l3 = Symbol('l3')
  #X = Matrix([x3,y3,0,l3])
  X3 = Matrix([x3,y3,0,1])

  x4 = Symbol('x4')
  y4 = Symbol('y4')
  l4 = Symbol('l4')
  #X4 = Matrix([x4,y4,0,l4])
  X4 = Matrix([x4,y4,0,1])


  #Project Points into image coordinates and normalize

  U1 = P*X1
  u1 = U1[0]/U1[2]
  v1 = U1[1]/U1[2]
  w1 = U1[2]/U1[2]

  U2 = P*X2
  u2 = U2[0]/U2[2]
  v2 = U2[1]/U2[2]
  w2 = U2[2]/U2[2]

  U3 = P*X3
  u3 = U3[0]/U3[2]
  v3 = U3[1]/U3[2]
  w3 = U3[2]/U3[2]

  U4 = P*X4
  u4 = U4[0]/U4[2]
  v4 = U4[1]/U4[2]
  w4 = U4[2]/U4[2]

  #      X = x1[:,i].T
  #      x = x2[0,i]
  #      y = x2[1,i]
  #      w = x2[2,i]
  #      Ades[2*i,:] = np.array([O, -w*X, y*X]).reshape(1, 9)
  #      Ades[2*i+1,:] = np.array([w*X, O, -x*X]).reshape(1, 9)

  #Asymb = Matrix([[   0,    0,     0, -w1*x1, -w1*y1, -w1*l1,  v1*x1,  v1*y1,  v1*l1],
  #                [w1*x1, w1*y1, w1*l1,      0,      0,      0, -u1*x1, -u1*y1, -u1*l1],
  #
  #                [   0,    0,     0, -w2*x2, -w2*y2, -w2*l2,  v2*x2,  v2*y2,  v2*l2],
  #                [w2*x2, w2*y2, w2*l2,      0,      0,      0, -u2*x2, -u2*y2, -u2*l2],
  #
  #                [   0,    0,     0, -w3*x3, -w3*y3, -w3*l3,  v3*x3,  v3*y3,  v3*l3],
  #                [w3*x3, w3*y3, w3*l3,      0,      0,      0, -u3*x3, -u3*y3, -u3*l3],
  #
  #                [   0,    0,     0, -w4*x4, -w4*y4, -w4*l4,  v4*x4,  v4*y4,  v4*l4],
  #                [w4*x4, w4*y4, w4*l4,      0,      0,      0, -u4*x4, -u4*y4, -u4*l4],
  #
  #                [0, 0, 0,      0,      0,      0, 0, 0, 0],
  #        ])


  # If we assume that object and image coordinates are normalized we can remove w and l from equations
  Asymb = Matrix([[   0,    0,     0, -x1, -y1, -1,  v1*x1,  v1*y1,  v1],
                  [x1, y1, 1,      0,      0,      0, -u1*x1, -u1*y1, -u1],

                  [   0,    0,     0, -x2, -y2, -1,  v2*x2,  v2*y2,  v2],
                  [x2, y2, 1,      0,      0,      0, -u2*x2, -u2*y2, -u2],

                  [   0,    0,     0, -x3, -y3, -1,  v3*x3,  v3*y3,  v3],
                  [x3, y3, 1,      0,      0,      0, -u3*x3, -u3*y3, -u3],

                  [   0,    0,     0, -x4, -y4, -1,  v4*x4,  v4*y4,  v4],
                  [x4, y4, 1,      0,      0,      0, -u4*x4, -u4*y4, -u4],
          ])
  return Asymb



def volker_metric_symb(A):
  # nomarlize each row
  for i in range(A.shape[0]):
    squared_sum = 0
    for j in range(A.shape[1]):
      squared_sum += sqrt(A[i,j]**2)
    A[i,:] = A[i,:] / squared_sum

  # compute the dot product
  As = A*A.T

  # we are interested only on the upper top triangular matrix coefficients
#  metric = 0
#  start = 1
#  for i in range(As.shape[0]):
#    for j in range(start,As.shape[0]):
#      metric = metric + As[i,j]**2
#    start += 1
  #metric = np.sum(As[[0,2,4,6],[1,3,5,7]]**2)

  #X vs X
  metric = np.sum(As[[0,0,0,2,2,4],[2,4,6,4,6,6]]**2)

  #Y vs Y
  metric = metric + np.sum(As[[1,1,1,3,3,5],[3,5,7,5,7,7]]**2)
  return  As, metric

def calculate_der_symb(metric):
  gradient = SymbGradient()

  x1 = Symbol('x1')
  y1 = Symbol('y1')
  l1 = Symbol('l1')


  x2 = Symbol('x2')
  y2 = Symbol('y2')
  l2 = Symbol('l2')


  x3 = Symbol('x3')
  y3 = Symbol('y3')
  l3 = Symbol('l3')


  x4 = Symbol('x4')
  y4 = Symbol('y4')
  l4 = Symbol('l4')

  gradient.d_x1 = diff(metric,x1)
  gradient.d_y1 = diff(metric,y1)

  gradient.d_x2 = diff(metric,x2)
  gradient.d_y2 = diff(metric,y2)

  gradient.d_x3 = diff(metric,x3)
  gradient.d_y3 = diff(metric,y3)

  gradient.d_x4 = diff(metric,x4)
  gradient.d_y4 = diff(metric,y4)
  return gradient

def evaluate_derivatives(gradient,objectPoints):
  gradient.d_x1_eval = evaluate_model_points(gradient.d_x1, objectPoints)
  gradient.d_y1_eval = evaluate_model_points(gradient.d_y1, objectPoints)

  gradient.d_x2_eval = evaluate_model_points(gradient.d_x2, objectPoints)
  gradient.d_y2_eval = evaluate_model_points(gradient.d_y2, objectPoints)

  gradient.d_x3_eval = evaluate_model_points(gradient.d_x3, objectPoints)
  gradient.d_y3_eval = evaluate_model_points(gradient.d_y3, objectPoints)

  gradient.d_x4_eval = evaluate_model_points(gradient.d_x4, objectPoints)
  gradient.d_y4_eval = evaluate_model_points(gradient.d_y4, objectPoints)

  return gradient

def update_points(alpha, gradient, objectPoints, limit=0.15):
  op = np.copy(objectPoints)
  op[0,0] += - gradient.d_x1_eval*alpha
  op[1,0] += - gradient.d_y1_eval*alpha

  op[0,1] += - gradient.d_x2_eval*alpha
  op[1,1] += - gradient.d_y2_eval*alpha

  op[0,2] += - gradient.d_x3_eval*alpha
  op[1,2] += - gradient.d_y3_eval*alpha

  op[0,3] += - gradient.d_x4_eval*alpha
  op[1,3] += - gradient.d_y4_eval*alpha

  op[0:3,:] = np.clip(op[0:3,:], -limit, limit)
  return op


def test_A_symb():
  Asymb = create_A_symb(cam.P)
  #Asquared, metric = volker_metric_symb(Asymb)

  objectPoints = np.copy(objectPoints_des)
  Atest = evaluate_model_points(Asymb, objectPoints)

  Xo = np.copy(objectPoints_des[[0,1,3],:]) #without the z coordinate (plane)
  Xi = np.copy(imagePoints_des)
  Atrue = calculate_A_matrix(Xo, Xi)

  print np.allclose(Atrue,Atest)

def test_Asquared_symb():
  objectPoints = np.copy(objectPoints_des)

  Asymb = create_A_symb(cam.P)
  Asquared_symb, volkerMetric_symb = volker_metric_symb(Asymb)



  volkerMetric_symb_eval = evaluate_model_points(volkerMetric_symb, objectPoints)



  Xo = np.copy(objectPoints_des[[0,1,3],:]) #without the z coordinate (plane)
  Xi = np.copy(imagePoints_des)
  Atrue = calculate_A_matrix(Xo, Xi)

  Asquared_true, volkerMetric = volker_metric(Atrue)

  Asquared_symb_eval = evaluate_model_points(Asquared_symb, objectPoints)

  print Asquared_true[0,0],Asquared_symb_eval[0,0]
  print volkerMetric,volkerMetric_symb_eval

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

#pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.3,0.3), n = (2,2))
#pl.random(n =4, r = 0.01, min_sep = 0.01)





Asymb = create_A_symb(cam.P)
Asquared_symb, volkerMetric_symb = volker_metric_symb(Asymb)
gradient = calculate_der_symb(volkerMetric_symb)

#%%
objectPoints_des = pl.get_points()
imagePoints_des = np.array(cam.project(objectPoints_des, False))
objectPoints_list = list()
imagePoints_list = list()
new_objectPoints = objectPoints_des
for i in range(100):
  objectPoints = np.copy(new_objectPoints)
  gradient = evaluate_derivatives(gradient,objectPoints)
  alpha = 0.0005
  new_objectPoints = update_points(alpha, gradient, objectPoints)
  new_imagePoints = np.array(cam.project(new_objectPoints, False))

  objectPoints_list.append(new_objectPoints)
  imagePoints_list.append(new_imagePoints)
  plt.ion()
  #plt.cla()
  plt.figure('Image Points')
  cam.plot_plane(pl)
  plt.plot(new_imagePoints[0],new_imagePoints[1],'-.',color = 'blue',)
  plt.plot(imagePoints_des[0],imagePoints_des[1],'x',color = 'black',)
  plt.xlim(0,1280)
  plt.ylim(0,960)
  plt.gca().invert_yaxis()
  plt.pause(0.01)

  print "Iteration: ", i
  print "dx1,dy1 :", gradient.d_x1_eval,gradient.d_y1_eval
  print "dx2,dy2 :", gradient.d_x2_eval,gradient.d_y2_eval
  print "dx3,dy3 :", gradient.d_x3_eval,gradient.d_y3_eval
  print "dx4,dy4 :", gradient.d_x4_eval,gradient.d_y4_eval
  print "------------------------------------------------------"


#plt.plot(imagePoints2[0],imagePoints2[1],'.',color = 'g',)
#plt.plot(imagePoints_ref[0],imagePoints_ref[1],'.',color = 'black',)
#plt.pause(0.05)
#Asymb = create_A_symb(cam.P)
#test_A_symb()

#test_Asquared_symb()
#Asquared, metric = volker_metric_symb(Asymb)


#print evaluate_A(Asquared, objectPoints)
#
#
#gradient = calculate_der_symb(metric)
#
#
##Evaluate derivatives points
#gradient = evaluate_derivatives(gradient,objectPoints)
#
#
##update gradient
#
#new_objectPoints = update_points(objectPoints)
#
#
#imagePoints2 = np.array(cam.project(op, False))
#
#plt.plot(imagePoints2[0],imagePoints2[1],'.',color = 'g',)
#plt.plot(imagePoints_ref[0],imagePoints_ref[1],'.',color = 'black',)
#plt.pause(0.05)


def dot_product(a):
  a_t = a.T
  result = zeros(a.rows, a.rows)

  for i in range(a.rows):
    for j in range(a.rows):
      result[i,j] = a.row(i).dot(a_t.col(j))
  return result



