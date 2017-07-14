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
    pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.6,0.6), n = (3,3))
    pl.uniform()
    #pl.update_random(n = 16, r = 0.05, min_sep = 0.05)

    objectPoints = pl.get_points()


    #create_uniform_cam_poses()
    #run_point_distribution_test(cam, objectPoints, False)

    iters = 1

    n_range = range(0,10)

    ippe_tvec_error_avg1 = list()
    ippe_rmat_error_avg1 = list()
    pnp_tvec_error_avg1 = list()
    pnp_rmat_error_avg1 = list()

    for n in n_range:
      ippe_tvec_error_sum = 0
      ippe_rmat_error_sum = 0
      pnp_tvec_error_sum = 0
      pnp_rmat_error_sum = 0
      for m in range(iters):
        ippe_tvec_error, ippe_rmat_error, pnp_tvec_error, pnp_rmat_error = run_single(cam, objectPoints, noise = n, quant_error = True, plot = False, debug = False)
        ippe_tvec_error_sum += ippe_tvec_error
        ippe_rmat_error_sum +=ippe_rmat_error
        pnp_tvec_error_sum += pnp_tvec_error
        pnp_rmat_error_sum += pnp_rmat_error
      ippe_tvec_error_avg1.append(ippe_tvec_error_sum/iters)
      ippe_rmat_error_avg1.append(ippe_rmat_error_sum/iters)
      pnp_tvec_error_avg1.append(pnp_tvec_error_sum/iters)
      pnp_rmat_error_avg1.append(pnp_rmat_error_sum/iters)




    #%%
    pl.uniform_with_distortion(mean = 0, sd = 0.1)
    objectPoints1 = pl.get_points()


    ippe_tvec_error_avg2 = list()
    ippe_rmat_error_avg2 = list()
    pnp_tvec_error_avg2 = list()
    pnp_rmat_error_avg2 = list()

    for n in n_range:
      ippe_tvec_error_sum = 0
      ippe_rmat_error_sum = 0
      pnp_tvec_error_sum = 0
      pnp_rmat_error_sum = 0
      for m in range(iters):
        ippe_tvec_error, ippe_rmat_error, pnp_tvec_error, pnp_rmat_error = run_single(cam, objectPoints1, noise = n, quant_error = False, plot = False, debug = False)
        ippe_tvec_error_sum += ippe_tvec_error
        ippe_rmat_error_sum +=ippe_rmat_error
        pnp_tvec_error_sum += pnp_tvec_error
        pnp_rmat_error_sum += pnp_rmat_error
      ippe_tvec_error_avg2.append(ippe_tvec_error_sum/iters)
      ippe_rmat_error_avg2.append(ippe_rmat_error_sum/iters)
      pnp_tvec_error_avg2.append(pnp_tvec_error_sum/iters)
      pnp_rmat_error_avg2.append(pnp_rmat_error_sum/iters)
#%%

    plt.figure()
    plt.title("Effect of the distribution of points on the translation estimation")
    plt.xlabel("Noise level (pixels)")
    plt.ylabel("percent error in t")
    plt.plot(n_range, ippe_tvec_error_avg1, label = "ippe uniform points")
    plt.plot(n_range, ippe_tvec_error_avg2, label = "ippe distorted points")
    plt.plot(n_range, pnp_tvec_error_avg1, label = "solvepnp uniform points")
    plt.plot(n_range, pnp_tvec_error_avg2, label = "solvepnp distorted points")
    plt.legend()

    plt.figure()
    plt.title("Effect of the distribution of points on the rotation estimation")
    plt.xlabel("Noise level (pixels)")
    plt.ylabel("rotation angle error (degrees)")
    plt.plot(n_range, ippe_rmat_error_avg1, label = "ippe uniform points")
    plt.plot(n_range, ippe_rmat_error_avg2, label = "ippe distorted points")
    plt.plot(n_range, pnp_rmat_error_avg1, label = "solvepnp uniform points")
    plt.plot(n_range, pnp_rmat_error_avg2, label = "solvepnp distorted points")
    plt.legend()


    plt.figure()
    imagePoints = np.array(cam.project(objectPoints,quant_error = False))
    cam.plot_image(imagePoints, points_color = 'blue')

    plt.figure()
    plt.plot(objectPoints[0,:], objectPoints[1,:],'o')
    plt.xlim(-0.8,0.8)
    plt.ylim(-0.8,0.8)

    plt.plot(objectPoints1[0,:], objectPoints1[1,:],'ro')
    plt.xlim(-0.8,0.8)
    plt.ylim(-0.8,0.8)










