# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:44:51 2017

@author: Raul Acuña
"""


#from math import sqrt


#from mpl_toolkits.mplot3d import Axes3D


from pose_sim import *


if __name__ == '__main__':
    #camera position in world coordinates
    x = 0.
    y = -0.8
    z = 2.

    # Create a camera
    cam = Camera()
    cam.set_K(fx = 800., fy = 800., cx = 320., cy = 240.)  #Camera Matrix
    cam.img_width = 320.*2.
    cam.img_height = 240.*2.
    #Camera looking straight down to the world center
    cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(160.0))
    #World position is defined after rotation matrix
    cam.set_world_position(x,y,z)
    cam.set_P() # create projection matrix

    #Create a plane with 4 points
    pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.6,0.6), n = (8,8))
    pl.uniform()
    #pl.update_random(n = 16, r = 0.05, min_sep = 0.05)

    objectPoints = pl.get_points()


    #create_uniform_cam_poses()
    #run_point_distribution_test(cam, objectPoints, False)

    iters = 5000

    n_range = np.arange(0,0.2,0.01)

    ippe_tvec_error_avg = list()
    ippe_rmat_error_avg = list()
    pnp_tvec_error_avg = list()
    pnp_rmat_error_avg = list()

    for n in n_range:
      ippe_tvec_error_sum = 0
      ippe_rmat_error_sum = 0
      pnp_tvec_error_sum = 0
      pnp_rmat_error_sum = 0
      for m in range(iters):
        if n==0:
          pl.uniform()
        else:
          pl.uniform_with_distortion(mean = 0, sd = n)
        objectPoints = pl.get_points()
        ippe_tvec_error, ippe_rmat_error, pnp_tvec_error, pnp_rmat_error = run_single(cam, objectPoints, noise = 0, quant_error = True, plot = False, debug = False)
        ippe_tvec_error_sum += ippe_tvec_error
        ippe_rmat_error_sum +=ippe_rmat_error
        pnp_tvec_error_sum += pnp_tvec_error
        pnp_rmat_error_sum += pnp_rmat_error
      ippe_tvec_error_avg.append(ippe_tvec_error_sum/iters)
      ippe_rmat_error_avg.append(ippe_rmat_error_sum/iters)
      pnp_tvec_error_avg.append(pnp_tvec_error_sum/iters)
      pnp_rmat_error_avg.append(pnp_rmat_error_sum/iters)


#%%

    plt.figure()
    plt.title("Effect of the distribution of points on the translation estimation")
    plt.xlabel("Amount of deviation from uniform pattern (noise pixels)")
    plt.ylabel("percent error in t")
    plt.plot(n_range, ippe_tvec_error_avg, label = "ippe")
    plt.plot(n_range, pnp_tvec_error_avg, label = "solvepnp")
    plt.legend()

    plt.figure()
    plt.title("Effect of the distribution of points on the rotation estimation")
    plt.xlabel("Amount of deviation from uniform pattern (noise pixels)")
    plt.ylabel("rotation angle error (degrees)")
    plt.plot(n_range, ippe_rmat_error_avg, label = "ippe")
    plt.plot(n_range, pnp_rmat_error_avg, label = "solvepnp")
    plt.legend()


    plt.figure()

    imagePoints = np.array(cam.project(objectPoints,quant_error = False))
    cam.plot_image(imagePoints, points_color = 'blue')

#%%
    plt.figure()
    plt.title("Model plane points and deformation for n=0.1 (uniform)")
    pl.uniform()
    objectPoints = pl.get_points()
    plt.plot(objectPoints[0,:], objectPoints[1,:],'o')
    plt.xlim(-0.8,0.8)
    plt.ylim(-0.8,0.8)


    pl.uniform_with_distortion(mean = 0, sd = 0.1)
    objectPoints = pl.get_points()
    plt.plot(objectPoints[0,:], objectPoints[1,:],'rx')











