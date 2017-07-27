#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:18:20 2017

@author: lracuna
"""

from pose_sim import *
from mpl_toolkits.mplot3d import Axes3D
from Rt_matrix_from_euler_t import R_matrix_from_euler_t
import itertools
from multiprocessing import Pool
from uniform_sphere import uniform_sphere

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

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  origin = np.zeros_like(unit_rays[0,:])
  #ax.quiver(origin,origin,origin,unit_rays[0,:],unit_rays[1,:],unit_rays[2,:], length=1.0, pivot = 'tail')
  #ax.scatter(unit_rays[0,:],unit_rays[1,:],unit_rays[2,:], 'rx')

  #Select a linear space of distances based on focal length (like in IPPE paper)
  #d_space = np.linspace(f/2,2*f, 4)

  #Select a linear space of distances based on focal length (like in IPPE paper)
  d_space = np.linspace(0.5,2.0,5)

  #t = d*unit_rays;
  t_list = []
  for d in d_space:

      #t_list.append(d*unit_rays)

      xx, yy, zz = uniform_sphere((0,360), (0,10), d, 10,5, False)

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

  ax.scatter(t_space[0,:],t_space[1,:],t_space[2,:], color = 'b')

  for pl in pl_space:
    objectPoints = pl.get_points()
    ax.scatter(objectPoints[0,:],objectPoints[1,:],objectPoints[2,:], color = 'r')

  i = 0
  ippe_tvec_error_sum = 0
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
    if plot:
      imagePoints = cam.project(objectPoints)
    cam.plot_image(imagePoints)
    if ((imagePoints[0,:]<cam.img_width) & (imagePoints[0,:]>0)).all():
      if ((imagePoints[1,:]<cam.img_height) & (imagePoints[1,:]>0)).all():
        cams.append(cam)


  if plot:
    planes = []
    pl.uniform()
    planes.append(pl)
    plot3D(cams, planes)

  return cams



if __name__ == '__main__':
    #camera position in world coordinates
    x = 0.
    y = 0.
    z = 2.

    # Create a camera
    cam = Camera()
    cam.set_K(fx = 800., fy = 800., cx = 320., cy = 240.)  #Camera Matrix
    cam.img_width = 320.*2.
    cam.img_height = 240.*2.
    #Camera looking straight down to the world center
    #cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180.0))
    #World position is defined after rotation matrix
    cam.set_world_position(x,y,z)
    cam.set_P() # create projection matrix

    #We Create a plane with 4 points with a uniform distribution
    pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.1,0.1), n = (2,2))

    max_deviation = 0.06
    deviation_range = np.arange(0,max_deviation+0.01,0.01)

    #Now we define a distribution of cameras on the space based on this plane
    #An optional paremeter is de possible deviation from uniform points
    cams = create_cam_distribution(cam, pl, max_deviation, plot=True)
#%%

    #We create the set of objectpoints from the planes as an array
    # for pool workers
    iters = 1



    pool = Pool()


    noise_range = [0, 1, 2, 3, 4]

    ippe_tvec_error_avg = list()
    ippe_rmat_error_avg = list()
    pnp_tvec_error_avg = list()
    pnp_rmat_error_avg = list()

    for i,noise in enumerate(noise_range):
      ippe_tvec_error_avg.append(list())
      ippe_rmat_error_avg.append(list())
      pnp_tvec_error_avg.append(list())
      pnp_rmat_error_avg.append(list())
      for deviation in deviation_range:
        print "Deviation", deviation
        ippe_tvec_error_sum = 0
        ippe_rmat_error_sum = 0
        pnp_tvec_error_sum = 0
        pnp_rmat_error_sum = 0
        for m in range(iters):
          if deviation==0:
            pl.uniform()
          else:
            pl.uniform_with_distortion(mean = 0, sd = deviation)
          objectPoints = pl.get_points()
#          for cam in cams:
#            ippe_tvec_error, ippe_rmat_error, pnp_tvec_error, pnp_rmat_error = run_single(cam, objectPoints, noise = noise, quant_error = False, plot = False, debug = False)
#            ippe_tvec_error_sum += ippe_tvec_error
#            ippe_rmat_error_sum +=ippe_rmat_error
#            pnp_tvec_error_sum += pnp_tvec_error
#            pnp_rmat_error_sum += pnp_rmat_error
          ret_values = np.array(pool.map(run_single_wrapper, itertools.izip(cams, itertools.repeat(objectPoints), itertools.repeat(noise))))
          ret_values_sum = np.sum(ret_values, axis=0)
          ippe_tvec_error_sum += ret_values_sum[0]
          ippe_rmat_error_sum += ret_values_sum[1]
          pnp_tvec_error_sum += ret_values_sum[2]
          pnp_rmat_error_sum += ret_values_sum[3]
        ippe_tvec_error_avg[i].append(ippe_tvec_error_sum/(iters*len(cams)))
        ippe_rmat_error_avg[i].append(ippe_rmat_error_sum/(iters*len(cams)))
        pnp_tvec_error_avg[i].append(pnp_tvec_error_sum/(iters*len(cams)))
        pnp_rmat_error_avg[i].append(pnp_rmat_error_sum/(iters*len(cams)))




  #%%

    plt.figure("FigA")
    plt.title("Effect of the distribution of points on the translation estimation")
    plt.xlabel("Gaussian standard deviation from uniform pattern (meters)")
    plt.ylabel("percent error in t")
    for i,noise in enumerate(noise_range):
      plt.plot(deviation_range, ippe_tvec_error_avg[i], label = "ippe with "+str(noise)+"px noise")
      #plt.plot(deviation_range, pnp_tvec_error_avg[i] - pnp_tvec_error_avg[i][0], label = "solvepnp "+str(noise)+"px noise")
    plt.legend(loc='upper left')

    plt.figure("FigB")
    plt.title("Effect of the distribution of points on the rotation estimation")
    plt.xlabel("Gaussian standard deviation from uniform pattern (meters)")
    plt.ylabel("rotation angle error (degrees)")
    for i,noise in enumerate(noise_range):
      plt.plot(deviation_range, ippe_rmat_error_avg[i], label = "ippe with "+str(noise)+"px noise")
      #plt.plot(deviation_range, pnp_rmat_error_avg[i] - pnp_rmat_error_avg[i][0], label = "solvepnp "+str(noise)+"px noise")
    plt.legend(loc='upper left')


#    plt.figure()
#
#    imagePoints = np.array(cam.project(objectPoints,quant_error = False))
#    cam.plot_image(imagePoints, points_color = 'blue')
#
#  #%%
#    plt.figure()
#    plt.title("Model plane points and deformation for n=0.1 (uniform)")
#    pl.uniform()
#    objectPoints = pl.get_points()
#    plt.plot(objectPoints[0,:], objectPoints[1,:],'o')
#    plt.xlim(-0.8,0.8)
#    plt.ylim(-0.8,0.8)
#
#
#    pl.uniform_with_distortion(mean = 0, sd = 0.1)
#    objectPoints = pl.get_points()
#    plt.plot(objectPoints[0,:], objectPoints[1,:],'rx')
