#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:52:11 2017

@author: lracuna
"""
from vision.camera import *
from vision.plane import Plane
import autograd.numpy as np
from autograd import grad
from vision.error_functions import geometric_distance_points, get_matrix_conditioning_number, volker_metric,calculate_A_matrix

class Gradient(object):
  def __init__(self):
    self.dx1 = None
    self.dy1 = None
    self.dx2 = None
    self.dy2 = None
    self.dx3 = None
    self.dy3 = None
    self.dx4 = None
    self.dy4 = None

    self.dx1_eval = 0
    self.dy1_eval = 0

    self.dx2_eval = 0
    self.dy2_eval = 0

    self.dx3_eval = 0
    self.dy3_eval = 0

    self.dx4_eval = 0
    self.dy4_eval = 0

    self.dx1_eval_old = 0
    self.dy1_eval_old = 0

    self.dx2_eval_old = 0
    self.dy2_eval_old = 0

    self.dx3_eval_old = 0
    self.dy3_eval_old = 0

    self.dx4_eval_old = 0
    self.dy4_eval_old = 0

    self.n = None #step in gradient descent

    self.n_x1 = self.n
    self.n_x2 = self.n
    self.n_x3 = self.n
    self.n_x4 = self.n

    self.n_y1 = self.n
    self.n_y2 = self.n
    self.n_y3 = self.n
    self.n_y4 = self.n

  def set_n(self,n):
    #n = 0.000001 #condition number norm 4 points
    self.n = n
    self.n_pos = 0.02*n # for SuperSAB
    self.n_neg = 0.03*n # for SuperSAB
    self.n_x1 = n
    self.n_x2 = n
    self.n_x3 = n
    self.n_x4 = n

    self.n_y1 = n
    self.n_y2 = n
    self.n_y3 = n
    self.n_y4 = n



def calculate_A_matrix_autograd(x1,y1,x2,y2,x3,y3,x4,y4,P,normalize=False):
  """ Calculate the A matrix for the DLT algorithm:  A.H = 0
  all coordinates are in object plane
  """
  X1 = np.array([[x1],[y1],[0.],[1.]]).reshape(4,1)
  X2 = np.array([[x2],[y2],[0.],[1.]]).reshape(4,1)
  X3 = np.array([[x3],[y3],[0.],[1.]]).reshape(4,1)
  X4 = np.array([[x4],[y4],[0.],[1.]]).reshape(4,1)

  U1 = np.array(np.dot(P,X1)).reshape(3,1)
  U2 = np.array(np.dot(P,X2)).reshape(3,1)
  U3 = np.array(np.dot(P,X3)).reshape(3,1)
  U4 = np.array(np.dot(P,X4)).reshape(3,1)

  object_pts = np.hstack([X1,X2,X3,X4])
  image_pts = np.hstack([U1,U2,U3,U4])

  if normalize:
    object_pts_norm,T1 = normalise_points(object_pts)
    image_pts_norm,T2 = normalise_points(image_pts)
  else:
    object_pts_norm = object_pts[[0,1,3],:]
    image_pts_norm = image_pts

  x1 = object_pts_norm[0,0]/object_pts_norm[2,0]
  y1 = object_pts_norm[1,0]/object_pts_norm[2,0]

  x2 = object_pts_norm[0,1]/object_pts_norm[2,1]
  y2 = object_pts_norm[1,1]/object_pts_norm[2,1]

  x3 = object_pts_norm[0,2]/object_pts_norm[2,2]
  y3 = object_pts_norm[1,2]/object_pts_norm[2,2]

  x4 = object_pts_norm[0,3]/object_pts_norm[2,3]
  y4 = object_pts_norm[1,3]/object_pts_norm[2,3]


  u1 = image_pts_norm[0,0]/image_pts_norm[2,0]
  v1 = image_pts_norm[1,0]/image_pts_norm[2,0]

  u2 = image_pts_norm[0,1]/image_pts_norm[2,1]
  v2 = image_pts_norm[1,1]/image_pts_norm[2,1]

  u3 = image_pts_norm[0,2]/image_pts_norm[2,2]
  v3 = image_pts_norm[1,2]/image_pts_norm[2,2]

  u4 = image_pts_norm[0,3]/image_pts_norm[2,3]
  v4 = image_pts_norm[1,3]/image_pts_norm[2,3]

  A = np.array([    [ 0,  0, 0, -x1, -y1, -1,  v1*x1,  v1*y1,  v1],
                    [x1, y1, 1,   0,   0,  0, -u1*x1, -u1*y1, -u1],

                    [ 0,  0, 0, -x2, -y2, -1,  v2*x2,  v2*y2,  v2],
                    [x2, y2, 1,   0,   0,  0, -u2*x2, -u2*y2, -u2],

                    [ 0,  0, 0, -x3, -y3, -1,  v3*x3,  v3*y3,  v3],
                    [x3, y3, 1,   0,   0,  0, -u3*x3, -u3*y3, -u3],

                    [0,   0, 0, -x4, -y4, -1,  v4*x4,  v4*y4,  v4],
                    [x4, y4, 1,   0,   0,  0, -u4*x4, -u4*y4, -u4],
          ])
  return A

# THIS FUNCTION DOESNT WORK WITH NORMALIZATION YET
def volker_metric_autograd(x1,y1,x2,y2,x3,y3,x4,y4,P):
  A = calculate_A_matrix_autograd(x1,y1,x2,y2,x3,y3,x4,y4,P)

  row_sums = list()
  for i in range(A.shape[0]):
    squared_sum = 0
    for j in range(A.shape[1]):
      squared_sum += np.sqrt(A[i,j]**2)
    row_sums.append(squared_sum)

  row_sums = np.array(row_sums).reshape(1,8)
  A = A/(row_sums.T)

  # compute the dot product
  As = np.dot(A,A.T)

  # we are interested only on the upper top triangular matrix coefficients
  metric = 0
  start = 1
  for i in range(As.shape[0]):
    for j in range(start,As.shape[0]):
      metric = metric +  As[i,j]**2
    start = start +1

  return  metric

# DONT USE PNORM
def matrix_pnorm_condition_number_autograd(x1,y1,x2,y2,x3,y3,x4,y4,P):
    A = calculate_A_matrix_autograd(x1,y1,x2,y2,x3,y3,x4,y4,P)
    A = np.conjugate(A)
    U, s, Vt = np.linalg.svd(A,0)
    m = U.shape[0]
    n = Vt.shape[1]
    rcond=1e-10
    cutoff = rcond*np.max(s)

    #    for i in range(min(n, m)):
    #        if s[i] > cutoff:
    #            s[i] = 1./s[i]
    #        else:
    #            s[i] = 0.
    new_s = list()
    for i in range(min(n, m)):
        if s[i] > cutoff:
            new_s.append(1./s[i])
        else:
            new_s.append(0.)
    new_s = np.array(new_s)
    pinv = np.dot(Vt.T, np.multiply(s[:, np.newaxis], U.T))
    #https://de.mathworks.com/help/symbolic/cond.html?requestedDomain=www.mathworks.com
    return np.linalg.norm(A)*np.linalg.norm(pinv)

def matrix_condition_number_autograd(x1,y1,x2,y2,x3,y3,x4,y4,P, image_pts_measured, normalize = False):
  A = calculate_A_matrix_autograd(x1,y1,x2,y2,x3,y3,x4,y4,P, normalize)

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

def repro_error_autograd(x1,y1,x2,y2,x3,y3,x4,y4,P, image_pts_measured, normalize = False):
  X1 = np.array([[x1],[y1],[0.],[1.]]).reshape(4,1)
  X2 = np.array([[x2],[y2],[0.],[1.]]).reshape(4,1)
  X3 = np.array([[x3],[y3],[0.],[1.]]).reshape(4,1)
  X4 = np.array([[x4],[y4],[0.],[1.]]).reshape(4,1)

  U1 = np.array(np.dot(P,X1)).reshape(3,1)
  U2 = np.array(np.dot(P,X2)).reshape(3,1)
  U3 = np.array(np.dot(P,X3)).reshape(3,1)
  U4 = np.array(np.dot(P,X4)).reshape(3,1)

  U1 = U1/U1[2,0]
  U2 = U2/U2[2,0]
  U3 = U3/U3[2,0]
  U4 = U4/U4[2,0]
#
#  u1_r = U1[0,0]/U1[2,0]
#  v1_r = U1[1,0]/U1[2,0]
#
#  u2_r = U2[0,1]/U2[2,1]
#  v2_r = U2[1,1]/U2[2,1]
#
#  u3_r = U3[0,2]/U3[2,2]
#  v3_r = U3[1,2]/U3[2,2]
#
#  u4_r = U4[0,3]/U4[2,3]
#  v4_r = U4[1,3]/U4[2,3]

  object_pts = np.hstack([X1,X2,X3,X4])
  image_pts_repro = np.hstack([U1,U2,U3,U4])

  x = image_pts_measured[:2,:]-image_pts_repro[:2,:]
  repro = np.sum(x**2)**(1./2)

  return repro
#  x1 = object_pts[0,0]/object_pts[2,0]
#  y1 = object_pts[1,0]/object_pts[2,0]
#
#  x2 = object_pts[0,1]/object_pts[2,1]
#  y2 = object_pts[1,1]/object_pts[2,1]
#
#  x3 = object_pts[0,2]/object_pts[2,2]
#  y3 = object_pts[1,2]/object_pts[2,2]
#
#  x4 = object_pts[0,3]/object_pts[2,3]
#  y4 = object_pts[1,3]/object_pts[2,3]
#
#
#  u1_r = image_pts_repro[0,0]/image_pts_repro[2,0]
#  v1_r = image_pts_repro[1,0]/image_pts_repro[2,0]
#
#  u2_r = image_pts_repro[0,1]/image_pts_repro[2,1]
#  v2_r = image_pts_repro[1,1]/image_pts_repro[2,1]
#
#  u3_r = image_pts_repro[0,2]/image_pts_repro[2,2]
#  v3_r = image_pts_repro[1,2]/image_pts_repro[2,2]
#
#  u4_r = image_pts_repro[0,3]/image_pts_repro[2,3]
#  v4_r = image_pts_repro[1,3]/image_pts_repro[2,3]


def hom_3d_to_2d(pts):
    pts = pts[[0,1,3],:]
    return pts

def hom_2d_to_3d(pts):
    pts = np.insert(pts,2,np.zeros(pts.shape[1]),0)
    return pts

def normalise_points(pts):
    """
    Function translates and normalises a set of 2D or 3d homogeneous points
    so that their centroid is at the origin and their mean distance from
    the origin is sqrt(2).  This process typically improves the
    conditioning of any equations used to solve homographies, fundamental
    matrices etc.


    Inputs:
    pts: 3xN array of 2D homogeneous coordinates

    Returns:
    newpts: 3xN array of transformed 2D homogeneous coordinates.  The
            scaling parameter is normalised to 1 unless the point is at
            infinity.
    T: The 3x3 transformation matrix, newpts = T*pts
    """
    if pts.shape[0] == 4:
        pts = hom_3d_to_2d(pts)

    if pts.shape[0] != 3 and pts.shape[0] != 4  :
        print "Shape error"


    finiteind = np.nonzero(abs(pts[2,:]) > np.spacing(1))

    if len(finiteind[0]) != pts.shape[1]:
        print('Some points are at infinity')

    dist = []
    pts = pts/pts[2,:]
    for i in finiteind:
        #Replaced below for autograd
#        pts[0,i] = pts[0,i]/pts[2,i]
#        pts[1,i] = pts[1,i]/pts[2,i]
#        pts[2,i] = 1;

        c = np.mean(pts[0:2,i].T, axis=0).T

        newp1 = pts[0,i]-c[0]
        newp2 = pts[1,i]-c[1]

        dist.append(np.sqrt(newp1**2 + newp2**2))

    dist = np.array(dist)

    meandist = np.mean(dist)

    scale = np.sqrt(2)/meandist

    T = np.array([[scale, 0, -scale*c[0]], [0, scale, -scale*c[1]], [0, 0, 1]])

    newpts = np.dot(T,pts)


    return newpts, T

def create_gradient(metric='condition_number', n = 0.000001):
  """"
  metric: 'condition_number' (default)
          'volker_metric
  """
  if metric == 'condition_number':
    metric_function = matrix_condition_number_autograd
  elif metric == 'pnorm_condition_number':
    metric_function = matrix_pnorm_condition_number_autograd
  elif metric == 'volker_metric':
    metric_function = volker_metric_autograd
  elif metric == 'repro_error':
    metric_function = repro_error_autograd

  gradient = Gradient()
  gradient.set_n(n)
  gradient.dx1 = grad(metric_function,0)
  gradient.dy1 = grad(metric_function,1)

  gradient.dx2 = grad(metric_function,2)
  gradient.dy2 = grad(metric_function,3)

  gradient.dx3 = grad(metric_function,4)
  gradient.dy3 = grad(metric_function,5)

  gradient.dx4 = grad(metric_function,6)
  gradient.dy4 = grad(metric_function,7)
  return gradient


def extract_objectpoints_vars(objectPoints):
  x1 = objectPoints[0,0]
  y1 = objectPoints[1,0]

  x2 = objectPoints[0,1]
  y2 = objectPoints[1,1]

  x3 = objectPoints[0,2]
  y3 = objectPoints[1,2]

  x4 = objectPoints[0,3]
  y4 = objectPoints[1,3]

  return [x1,y1,x2,y2,x3,y3,x4,y4]

def evaluate_gradient(gradient, objectPoints, P, image_pts_measured, normalize = False, ):
  x1,y1,x2,y2,x3,y3,x4,y4 = extract_objectpoints_vars(objectPoints)

  gradient.dx1_eval_old = gradient.dx1_eval
  gradient.dy1_eval_old = gradient.dy1_eval

  gradient.dx2_eval_old = gradient.dx2_eval
  gradient.dy2_eval_old = gradient.dy2_eval

  gradient.dx3_eval_old = gradient.dx3_eval
  gradient.dy3_eval_old = gradient.dy3_eval

  gradient.dx4_eval_old = gradient.dx4_eval
  gradient.dy4_eval_old = gradient.dy4_eval


  gradient.dx1_eval = gradient.dx1(x1,y1,x2,y2,x3,y3,x4,y4, P, image_pts_measured, normalize)*gradient.n_x1
  gradient.dy1_eval = gradient.dy1(x1,y1,x2,y2,x3,y3,x4,y4, P, image_pts_measured, normalize)*gradient.n_y1

  gradient.dx2_eval = gradient.dx2(x1,y1,x2,y2,x3,y3,x4,y4, P, image_pts_measured, normalize )*gradient.n_x2
  gradient.dy2_eval = gradient.dy2(x1,y1,x2,y2,x3,y3,x4,y4, P, image_pts_measured, normalize)*gradient.n_y2

  gradient.dx3_eval = gradient.dx3(x1,y1,x2,y2,x3,y3,x4,y4, P, image_pts_measured, normalize)*gradient.n_x3
  gradient.dy3_eval = gradient.dy3(x1,y1,x2,y2,x3,y3,x4,y4, P, image_pts_measured, normalize)*gradient.n_y3

  gradient.dx4_eval = gradient.dx4(x1,y1,x2,y2,x3,y3,x4,y4, P, image_pts_measured, normalize)*gradient.n_x4
  gradient.dy4_eval = gradient.dy4(x1,y1,x2,y2,x3,y3,x4,y4, P, image_pts_measured ,normalize)*gradient.n_y4

  


  gradient.n_x1 = supersab(gradient.n_x1,gradient.dx1_eval,gradient.dx1_eval_old,gradient.n_pos,gradient.n_neg)
  gradient.n_x2 = supersab(gradient.n_x2,gradient.dx2_eval,gradient.dx2_eval_old,gradient.n_pos,gradient.n_neg)
  gradient.n_x3 = supersab(gradient.n_x3,gradient.dx3_eval,gradient.dx3_eval_old,gradient.n_pos,gradient.n_neg)
  gradient.n_x4 = supersab(gradient.n_x4,gradient.dx4_eval,gradient.dx4_eval_old,gradient.n_pos,gradient.n_neg)

  gradient.n_y1 = supersab(gradient.n_y1,gradient.dy1_eval,gradient.dy1_eval_old,gradient.n_pos,gradient.n_neg)
  gradient.n_y2 = supersab(gradient.n_y2,gradient.dy2_eval,gradient.dy2_eval_old,gradient.n_pos,gradient.n_neg)
  gradient.n_y3 = supersab(gradient.n_y3,gradient.dy3_eval,gradient.dy3_eval_old,gradient.n_pos,gradient.n_neg)
  gradient.n_y4 = supersab(gradient.n_y4,gradient.dy4_eval,gradient.dy4_eval_old,gradient.n_pos,gradient.n_neg)
  
  ## Limit
  limit = 1
  gradient.dx1_eval = np.clip(gradient.dx1_eval, -limit, limit)
  gradient.dy1_eval = np.clip(gradient.dy1_eval, -limit, limit)

  gradient.dx2_eval = np.clip(gradient.dx2_eval, -limit, limit)
  gradient.dy2_eval = np.clip(gradient.dy2_eval, -limit, limit)

  gradient.dx3_eval = np.clip(gradient.dx3_eval, -limit, limit)
  gradient.dy3_eval = np.clip(gradient.dy3_eval, -limit, limit)

  gradient.dx4_eval = np.clip(gradient.dx4_eval, -limit, limit)
  gradient.dy4_eval = np.clip(gradient.dy4_eval, -limit, limit)

  return gradient

def supersab(n, gradient_eval_current, gradient_eval_old, n_pos, n_neg):
  if np.sign(gradient_eval_current*gradient_eval_old) > 0:
    n = n + n_pos
  else:
    n = n - n_neg
  return n

def update_points(gradient, objectPoints, limitx=0.15,limity=0.15):
  op = np.copy(objectPoints)
  op[0,0] += - gradient.dx1_eval
  op[1,0] += - gradient.dy1_eval

  op[0,1] += - gradient.dx2_eval
  op[1,1] += - gradient.dy2_eval

  op[0,2] += - gradient.dx3_eval
  op[1,2] += - gradient.dy3_eval

  op[0,3] += - gradient.dx4_eval
  op[1,3] += - gradient.dy4_eval

  circle = True
  radius = 0.15
  if (circle):
      for i in range(op.shape[1]):
          distance = np.sqrt(op[0,i]**2+op[1,i]**2)
          if distance > radius:
              op[:3,i] = op[:3,i]*radius/distance

  else:
      op[0,:] = np.clip(op[0,:], -limitx, limitx)
      op[1,:] = np.clip(op[1,:], -limity, limity)
  return op