#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:21:26 2018

@author: lracuna
"""

import autograd.numpy as np
from optimize.utils import flatten_points, unflatten_points, normalise_points

def calculate_A_matrix_autograd(params,P,normalize=False):
  P = np.array(P)
  n_points = params.shape[0]/2

  object_pts = unflatten_points(params, type='object_plane')
  image_pts = np.array(np.dot(P,object_pts))
  image_pts = image_pts/image_pts[2,:]

  if normalize:
    object_pts_norm,T1 = normalise_points(object_pts)
    image_pts_norm,T2 = normalise_points(image_pts)
  else:
    object_pts_norm = object_pts[[0,1,3],:]
    image_pts_norm = image_pts

  object_pts_norm = object_pts_norm/object_pts_norm[2,:]
  image_pts_norm = image_pts_norm/image_pts_norm[2,:]

  A = np.array([]).reshape([0,9])
  for i in range(n_points):
      x = object_pts_norm[0,i]
      y = object_pts_norm[1,i]
      u = image_pts_norm[0,i]
      v = image_pts_norm[1,i]

      row1 = np.array([ 0,  0, 0, -x, -y, -1,  v*x,  v*y,  v])
      row2 = np.array([x, y, 1,   0,   0,  0, -u*x, -u*y, -u])

      A = np.vstack([A, row1])
      A = np.vstack([A, row2])
  return A


def matrix_condition_number_autograd(params,P,normalize = False):
  A = calculate_A_matrix_autograd(params,P, normalize)
  U, s, V = np.linalg.svd(A,full_matrices=False)
  greatest_singular_value = s[0]
#  rcond=1e-5
#  if s[-1] > rcond:
#    smalles_singular_value = s[-1]
#  else:
#    smalles_singular_value = s[-2]
  smallest_singular_value = s[-2]
  return greatest_singular_value/smallest_singular_value
  #return np.sqrt(greatest_singular_value)/np.sqrt(smallest_singular_value)