#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:37:32 2017

@author: lracuna
"""

from numpy import random, cos, sin, sqrt, pi, linspace, deg2rad, meshgrid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def uniform_sphere(theta_range, phi_range, r, n_theta = 10, n_phi = 10, plot = False):
  """n points distributed evenly on the surface of a unit sphere
  theta_range: tuple (min = 0,max = 360)
  phi_range: tuple (min =0,max =90)
  r: radius of the sphere
  n_theta: number of points in theta
  n_phi: number of points in phi

  """
  space_theta = linspace(deg2rad(theta_range[0]), deg2rad(theta_range[1]), n_theta)
  space_phi = linspace(deg2rad(phi_range[0]), deg2rad(phi_range[1]), n_phi)
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

x, y, z = uniform_sphere((0,360), (0,40), 10, 5,5, True)