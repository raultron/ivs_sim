#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:20:42 2018

@author: lracuna
"""
import autograd.numpy as np
def flatten_points(points, type = 'image'):
  """
  Points:
    Array of n points in homogeneous coordinates
    example image points: np.array([[u1,v1,1], [u2,v2,1] ,....,[un,vn,1] ])

  Type:
    image: 2d image points in homogeneous coordinates [u,v,1]
    object: 3d world points in homogeneous coordinates [x,y,z,1]
    object_plane: 3d world points on a plane in homogeneous coordinates [x,y,0,1]
  """
  if type == 'image':
    # normalize and remove the homogeneous term
    n_points = np.copy(points/points[2,:])
    n_points = n_points[:2,:]

  if type == 'object_plane':
    # normalize and remove the homogeneous term
    n_points = np.copy(points/points[3,:])
    n_points = n_points[:2,:]

  if type == 'object':
    # normalize and remove the homogeneous term
    n_points = np.copy(points/points[3,:])
    n_points = n_points[:3,:]
  out = n_points.flatten('F')

  return out


def unflatten_points(points, type = 'image'):
  """
  Points:
    Array vector of flatened n points.
    example image points: np.array([u1,v1, u2,v2,....,un,vn])

  Type: output format
    image: 2d image points in homogeneous coordinates [u,v,1]
    object: 3d world points in homogeneous coordinates [x,y,z,1]
    object_plane: 3d world points on a plane in homogeneous coordinates [x,y,0,1]
  """
  if type == 'image':
    out = points.reshape((2,-1),order = 'F')
    out = np.vstack([out, np.ones(out.shape[1])])

  if type == 'object_plane':
    out = points.reshape((2,-1),order = 'F')
    out = np.vstack([out, np.zeros(out.shape[1])])
    out = np.vstack([out, np.ones(out.shape[1])])

  if type == 'object':
    out = points.reshape((3,-1),order = 'F')
    out = np.vstack([out, np.ones(out.shape[1])])

  return out


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

def hom_3d_to_2d(pts):
    pts = pts[[0,1,3],:]
    return pts

def hom_2d_to_3d(pts):
    pts = np.insert(pts,2,np.zeros(pts.shape[1]),0)
    return pts