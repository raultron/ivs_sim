#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:37:32 2017

@author: lracuna
"""

from numpy import random, cos, sin, sqrt, pi, linspace, deg2rad, meshgrid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def uniform_sphere(theta_params = (0,360,10), phi_params = (0,90,10), r = 1., plot = False):
  """n points distributed evenly on the surface of a unit sphere
  theta_params: tuple (min = 0,max = 360, N divisions = 10)
  phi_params: tuple (min =0,max =90, N divisions = 10)
  r: radius of the sphere
  n_theta: number of points in theta
  n_phi: number of points in phi

  """
  
  space_theta = linspace(deg2rad(theta_params[0]), deg2rad(theta_params[1]), theta_params[2])
  print space_theta
  space_phi = linspace(deg2rad(phi_params[0]), deg2rad(phi_params[1]), phi_params[2])
  print space_phi
  theta, phi = meshgrid(space_theta,space_phi )

  x = r*cos(theta)*sin(phi)
  y = r*sin(theta)*sin(phi)
  z = r*cos(phi)
  if plot:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()

  return x, y, z

#x, y, z = uniform_sphere((0,360), (0,40), 10, 5,5, True)