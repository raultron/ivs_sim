# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:38:35 2017

@author: racuna
"""

import matplotlib.pyplot as plt
import autograd.numpy as np

from vision.camera import Camera
from vision.plane import Plane
from vision.rt_matrix import R_matrix_from_euler_t
from vision.plot_tools import plot3D


def uniform_sphere(theta_params = (0,360,10), phi_params = (0,90,10), r = 1., plot = False):
  """n points distributed evenly on the surface of a unit sphere
  theta_params: tuple (min = 0,max = 360, N divisions = 10)
  phi_params: tuple (min =0,max =90, N divisions = 10)
  r: radius of the sphere
  n_theta: number of points in theta
  n_phi: number of points in phi

  """
  space_theta = linspace(deg2rad(theta_params[0]), deg2rad(theta_params[1]), theta_params[2])
  space_phi = linspace(deg2rad(phi_params[0]), deg2rad(phi_params[1]), phi_params[2])
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

def create_cam_distribution(cam = None, plane_size = (0.3,0.3), theta_params = (0,360,10), phi_params =  (0,70,5), r_params = (0.25,1.0,4), plot=False):
  if cam == None:
    # Create an initial camera on the center of the world
    cam = Camera()
    f = 800
    cam.set_K(fx = f, fy = f, cx = 320, cy = 240)  #Camera Matrix
    cam.img_width = 320*2
    cam.img_height = 240*2

  # we create a default plane with 4 points with a side lenght of w (meters)
  plane =  Plane(origin=np.array([0, 0, 0] ), normal = np.array([0, 0, 1]), size=plane_size, n = (2,2))
  #We extend the size of this plane to account for the deviation from a uniform pattern
  #plane.size = (plane.size[0] + deviation, plane.size[1] + deviation)

  d_space = np.linspace(r_params[0],r_params[1],r_params[2])
  t_list = []
  for d in d_space:
      xx, yy, zz = uniform_sphere(theta_params, phi_params, d, False)
      sphere_points = np.array([xx.ravel(),yy.ravel(), zz.ravel()], dtype=np.float32)
      t_list.append(sphere_points)
  t_space = np.hstack(t_list)

  cams = []
  for t in t_space.T:
    cam = cam.clone()
    cam.set_t(-t[0], -t[1],-t[2])
    cam.set_R_mat(R_matrix_from_euler_t(0.0,0,0))
    cam.look_at([0,0,0])

    plane.set_origin(np.array([0, 0, 0]))
    plane.uniform()
    objectPoints = plane.get_points()
    imagePoints = cam.project(objectPoints)

    #if plot:
    #  cam.plot_image(imagePoints)
    if ((imagePoints[0,:]<cam.img_width) & (imagePoints[0,:]>0)).all():
      if ((imagePoints[1,:]<cam.img_height) & (imagePoints[1,:]>0)).all():
        cams.append(cam)

  if plot:
    planes = []
    plane.uniform()
    planes.append(plane)
    plot3D(cams, planes)

  return cams

if __name__ == "__main__":
    # Test the defined class
    create_cam_distribution(None, plane_size=(0.3, 0.3),
                            theta_params=(0, 360, 10), phi_params=(0, 70, 5),
                            r_params=(0.25, 1.0, 4), plot=True)
