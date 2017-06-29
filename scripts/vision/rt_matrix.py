#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:08:12 2017

@author: lracuna
"""
from scipy.linalg import expm
import numpy as np
def rotation_matrix(a, alpha):
    """  Creates a 3D [R|t] matrix for rotation
    around the axis of the vector a by an alpha angle.""" 
    #Normalize the rotation axis a
    a = a / np.linalg.norm(a)
    
    #Build the skew symetric
    a_skew = np.mat([[0,-a[2],a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(4)    
    R[:3,:3] = expm(a_skew*alpha)
    return R
  
def translation_matrix(t):
  """  Creates a 3D [R|t] matrix with a translation t
  and an identity rotation """
  R = np.eye(4)
  R[:3,3] = np.array([t[0],t[1],t[2]])
  return R

def rotation_matrix_from_two_vectors(a,b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a,b)
    ssc = np.matrix(np.array([[0.0, -v[2], v[1]],
                 [v[2], 0.0, -v[0]],
                 [-v[1], v[0], 0]]))
    R = np.eye(4)  
    R[:3,:3] = np.array(np.eye(3) + ssc + (ssc**2)*(1.0/(1.0+np.dot(a,b))))
    return R