# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:38:35 2017

@author: racuna
"""
from mayavi import mlab
#from scipy.linalg import expm, rq, det, inv
#import matplotlib.pyplot as plt
import autograd.numpy as np

#from math import atan
#from vision.rt_matrix import rotation_matrix
from numpy import random, cos, sin, sqrt, pi, linspace, deg2rad, meshgrid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from Rt_matrix_from_euler_t import R_matrix_from_euler_t


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

#x, y, z = uniform_sphere((0,360), (0,40), 10, 5,5, True)
def plot3D_cam(cam, axis_scale = 0.2):
    
    #Coordinate Frame of real camera
    #Camera axis
    cam_axis_x = np.array([1,0,0,1]).T
    cam_axis_y = np.array([0,1,0,1]).T
    cam_axis_z = np.array([0,0,1,1]).T

    cam_axis_x = np.dot(cam.R.T, cam_axis_x)
    cam_axis_y = np.dot(cam.R.T, cam_axis_y)
    cam_axis_z = np.dot(cam.R.T, cam_axis_z)

    cam_world = cam.get_world_position()

    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_x[0], cam_axis_x[1], cam_axis_x[2], line_width=3, scale_factor=axis_scale, color=(1-axis_scale,0,0))
    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_y[0], cam_axis_y[1], cam_axis_y[2], line_width=3, scale_factor=axis_scale, color=(0,1-axis_scale,0))
    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2], cam_axis_z[0], cam_axis_z[1], cam_axis_z[2], line_width=3, scale_factor=axis_scale, color=(0,0,1-axis_scale))


def plot3D(cams, planes):
    #mlab.figure(figure=None, bgcolor=(0.1,0.5,0.5), fgcolor=None, engine=None, size=(400, 350))
    axis_scale = 0.05
    for cam in cams:
        plot3D_cam(cam, axis_scale)
        #axis_scale = axis_scale - 0.1

    for plane in planes:
        #Plot plane points in 3D
        plane_points = plane.get_points()
        mlab.points3d(plane_points[0], plane_points[1], plane_points[2], scale_factor=0.05, color = plane.get_color())
        mlab.points3d(plane_points[0,0], plane_points[1,0], plane_points[2,0], scale_factor=0.05, color = (0.,0.,1.))

    mlab.show()

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
    #if plot:
    #  cam.plot_image(imagePoints)
    if ((imagePoints[0,:]<cam.img_width) & (imagePoints[0,:]>0)).all():
      if ((imagePoints[1,:]<cam.img_height) & (imagePoints[1,:]>0)).all():
        cams.append(cam)

  if plot:
    planes = []
    pl.uniform()
    planes.append(pl)
    plot3D(cams, planes)

  return cams