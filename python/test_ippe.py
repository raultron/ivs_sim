# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:44:51 2017

@author: Raul Acuña
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab

from vision.rt_matrix import rot_matrix_error
from vision.camera import Camera
from vision.plane import Plane
from ippe import homo2d, ippe
import cv2

def init():   
    #camera position in world coordinates
    x = 0.
    y = -0.5
    z = 4 
    
    # Create a camera    
    cam = Camera()    
    cam.set_K(fx = 800, fy = 800, cx = 320, cy = 240)  #Camera Matrix
    cam.img_width = 320*2
    cam.img_height = 240*2    
    #Camera looking straight down to the world center
    cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(170.0))    
    #World position is defined after rotation matrix
    cam.set_world_position(x,y,z)   
    cam.set_P() # create projection matrix
    
    #Create a plane with 4 points
    pl =  Plane(origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), size=(0.6,0.6), n = (10,10))
    pl.update()
    #pl.update_random(n = 10, r = 0.05, min_sep = 0.05)  
    return cam, pl

def addnoise_imagePoints(imagePoints, mean = 0, sd = 2):
    """ Add Gaussian noise to image points
    imagePoints: 3xn points in homogeneous pixel coordinates
    mean: zero mean
    sd: pixels of standard deviation 
    """
    gaussian_noise = np.random.normal(mean,sd,(2,imagePoints.shape[1]))
    imagePoints[:2,:] = imagePoints[:2,:] + gaussian_noise
    return imagePoints

def show_homo2d_normalization(imagePoints):
    imagePoints_normalized = homo2d.normalise2dpts(imagePoints)
    imagePoints_normalized = imagePoints_normalized[0]
    plt.figure()
    plt.plot(imagePoints_normalized[0],imagePoints_normalized[1],'.',color = 'red',)
    plt.plot(imagePoints_normalized[0,0],imagePoints_normalized[1,0],'.',color = 'blue')
    plt.gca().invert_yaxis()
    plt.show()




def pose_ippe_both(objectPoints, normalizedimagePoints):
    """ This function calculates the pose using the IPPE algorithm which 
    returns to possible poses. It returns both poses for comparison.
    
    objectPoints:  4xn homogeneous 3D object coordinates
    normalizedimagePoints: 3xn homogeneous normalized pixel coordinates
    """
    print("Starting ippe pose calculation")
    x1 = objectPoints[:3,:] # homogeneous 3D coordinates
    x2 = normalizedimagePoints[:2,:] 
    ippe_result = ippe.mat_run(x1,x2)
    
    print("ippe finished succesfully")
    
    if ippe_result['reprojError1'] <= ippe_result['reprojError2']:
        ippe_best = '1'
        ippe_worst = '2'
    else:
        ippe_best = '2'
        ippe_worst = '1'
    
    #convert back to homgeneous coordinates
    ippe_tvec1 = np.zeros(4)
    ippe_tvec2 = np.zeros(4)
    ippe_tvec1[3] = 1
    ippe_tvec2[3] = 1
    ippe_rmat1 = np.eye(4)
    ippe_rmat2 = np.eye(4)
    
    ippe_tvec1[:3] = ippe_result['t'+ippe_best]
    ippe_rmat1[:3,:3] = ippe_result['R'+ippe_best]
    
    ippe_tvec2[:3] = ippe_result['t'+ippe_worst]
    ippe_rmat2[:3,:3] = ippe_result['R'+ippe_worst]
    
#    #hack to correct sign (check why does it happen)
#    #a rotation matrix has determinant with value equal to 1
#    if np.linalg.det(ippe_rmat) < 0:
#        ippe_rmat[:3,2] = -ippe_rmat[:3,2]
        
    
    return ippe_tvec1,ippe_rmat1,ippe_tvec2,ippe_rmat2
    
    
    
def pose_ippe_best(objectPoints, normalizedimagePoints):
    """ This function calculates the pose using the IPPE algorithm which 
    returns to possible poses. The best pose is then selected based on 
    the reprojection error and that the objectPoints have to be in front of the
    camera in marker coordinates.
    
    objectPoints:  4xn homogeneous 3D object coordinates
    normalizedimagePoints: 3xn homogeneous normalized pixel coordinates
    """
    
    print("Starting ippe pose calculation")
    x1 = objectPoints[:3,:] # homogeneous 3D coordinates
    x2 = normalizedimagePoints[:2,:] 
    ippe_result = ippe.mat_run(x1,x2)
    
    print("ippe finished succesfully")
    
    if ippe_result['reprojError1'] <= ippe_result['reprojError2']:
        ippe_valid = '1'
    else:
        ippe_valid = '2'
    
    #convert back to homgeneous coordinates
    ippe_tvec = np.zeros(4)
    ippe_tvec[3] = 1
    ippe_rmat = np.eye(4)
    
    ippe_tvec[:3] = ippe_result['t'+ippe_valid]
    ippe_rmat[:3,:3] = ippe_result['R'+ippe_valid]
    
    #hack to correct sign (check why does it happen)
    #a rotation matrix has determinant with value equal to 1
    if np.linalg.det(ippe_rmat) < 0:
        ippe_rmat[:3,2] = -ippe_rmat[:3,2]
        
    
    return ippe_tvec,ippe_rmat

def pose_pnp(objectPoints, imagePoints, K):
    """ This function calculates the pose using the OpenCV solvePnP algorithm     
    objectPoints:  4xn homogeneous 3D object coordinates
    normalizedimagePoints: 3xn homogeneous normalized pixel coordinates
    K: Camera matrix
    """
    retval, rvec, tvec = cv2.solvePnP(objectPoints[:3,:].T,imagePoints[:2,:].T,K, (0))
    print("solvePnP finished succesfully", retval)    
    
    rmat, j = cv2.Rodrigues(rvec) 
    
    #convert back to homgeneous coordinates
    pnp_tvec = np.zeros(4)
    pnp_tvec[3] = 1
    pnp_rmat = np.eye(4) 
    pnp_tvec[:3] = tvec[:,0]
    pnp_rmat[:3,:3] = rmat
    
    return pnp_tvec,pnp_rmat



def calc_estimated_pose_error(tvec_ref, rmat_ref, tvec_est, rmat_est):
    # Translation error percentual
    tvec_error = np.linalg.norm(tvec_est[:3] - tvec_ref[:3])/np.linalg.norm(tvec_ref[:3])*100
    
    #tvec_error = np.sqrt((np.sum((tvec_est[:3]- tvec_ref[:3])**2))    
    
    #Rotation matrix error
    rmat_error = rot_matrix_error(rmat_ref,rmat_est, method = 'angle')    
    #rmat_error = rot_matrix_error(rmat_ref,rmat_est) 
    return tvec_error, rmat_error


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
    
    axis_scale = 0.4
    for cam in cams:
        plot3D_cam(cam, axis_scale)
        axis_scale = axis_scale - 0.1
        
    for plane in planes:        
        #Plot plane points in 3D
        plane_points = plane.get_points()    
        mlab.points3d(plane_points[0], plane_points[1], plane_points[2], scale_factor=0.05, color = pl.get_color())
        mlab.points3d(plane_points[0,0], plane_points[1,0], plane_points[2,0], scale_factor=0.05, color = (0.,0.,1.))
    
    mlab.show()
    
def run_single(cam, model_plane, plot = True):
    objectPoints = model_plane.get_points()
    objectPoints[2,:] = 0
    
    
    #Project points in camera
    imagePoints = np.array(cam.project(objectPoints,quant_error=False))
    
    #Add Gaussian noise in pixel coordinates
    #imagePoints = addnoise_imagePoints(imagePoints, mean = 0, sd = 1)
    
    #Show projected points
    if plot:
        cam.plot_image(imagePoints, model_plane.get_color())
    
    #Show the effect of the homography normalization    
    #show_homo2d_normalization(imagePoints)    
    
    #Calculate the pose using solvepnp and plot the image points
    pnp_tvec, pnp_rmat = pose_pnp(objectPoints, imagePoints, cam.K)    
    pnpCam = cam.clone_withPose(pnp_tvec, pnp_rmat)
    #pnpCam.plot_image(imagePoints, pl.get_color())
    
    #Calculate the pose using IPPE and plot the image points
    normalizedimagePoints = cam.get_normalized_pixel_coordinates(imagePoints)
    ippe_tvec, ippe_rmat = pose_ippe_best(objectPoints, normalizedimagePoints)    
    ippeCam = cam.clone_withPose(ippe_tvec, ippe_rmat)    
    #ippeCam.plot_image(imagePoints, pl.get_color())
    
    
    ippe_tvec1,ippe_rmat1,ippe_tvec2,ippe_rmat2 = pose_ippe_both(objectPoints, normalizedimagePoints)
    ippeCam1 = cam.clone_withPose(ippe_tvec1, ippe_rmat1)    
    ippeCam2 = cam.clone_withPose(ippe_tvec2, ippe_rmat2)   
    #ippeCam2.plot_image(imagePoints, pl.get_color())
    
    
    
    #Calculate errors
    pnp_tvec_error, pnp_rmat_error = calc_estimated_pose_error(cam.get_tvec(), cam.R, pnpCam.get_tvec(), pnp_rmat)
    ippe_tvec_error, ippe_rmat_error = calc_estimated_pose_error(cam.get_tvec(), cam.R, ippeCam.get_tvec(), ippe_rmat)
    
    # Print errors
    
    # IPPE pose estimation errors
    print ("----------------------------------------------------")
    print ("Translation Errors")
    print("IPPE    : %f" % ippe_tvec_error)
    print("solvePnP: %f" % pnp_tvec_error)
    
    # solvePnP pose estimation errors
    print ("----------------------------------------------------")
    print ("Rotation Errors")
    print("IPPE    : %f" % ippe_rmat_error)
    print("solvePnP: %f" % pnp_rmat_error)
    
    #cams = [cam, ippeCam1, ippeCam2, pnpCam]
    #planes = [pl]
    #plot3D(cams, planes)
    
    return ippe_tvec_error, ippe_rmat_error, pnp_tvec_error, pnp_rmat_error


def create_uniform_cam_poses():
    
    #%%
    # Create a camera on the center of the world
    cam = Camera()   
    f = 800
    cam.set_K(fx = f, fy = f, cx = 320, cy = 240)  #Camera Matrix
    cam.img_width = 320*2
    cam.img_height = 240*2
    
    
    # We create an uniform distribution of points in image coordinates
    
    x_min = cam.img_width/2 -10
    x_max = cam.img_width/2 + 10
    y_min = cam.img_height/2 -10
    y_max = cam.img_height/2 +10
    
    x_dist = np.linspace(x_min,x_max, 2)
    y_dist = np.linspace(y_min,y_max,2)

    xx, yy = np.meshgrid(x_dist, y_dist)
    
    hh = np.ones_like(xx, dtype=np.float32)    
    imagePoints = np.array([xx.ravel(),yy.ravel(), hh.ravel()], dtype=np.float32)
    
     
    # Backproject the pixels into rays (unit vector with the tail at the camera center)
    Kinv = np.linalg.inv(cam.K)
    unit_rays = np.array(np.dot(Kinv,imagePoints))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    origin = np.zeros_like(unit_rays[0,:])
    ax.quiver(origin,origin,origin,unit_rays[0,:],unit_rays[1,:],unit_rays[2,:], length=1.0, pivot = 'tail')
    ax.scatter(unit_rays[0,:],unit_rays[1,:],unit_rays[2,:], color = 'r')
    
    
    
    #Select a linear space of distances based on focal length (like in IPPE paper)
    d_space = np.linspace(f/2,2*f, 4)
    
    #Select a linear space of distances based on focal length (like in IPPE paper)
    #d_space = np.linspace(0.2,2.0, 10)
    
    #t = d*unit_rays;
    t_list = []
    for d in d_space:
        t_list.append(d*unit_rays) 
    t_space = np.hstack(t_list)
    
    
    
    #we now create a plane model for each t
    w = 200 #pixel units
    #w = 0.2 #meters
    pl_space = []
    for t in t_space.T:
        pl =  Plane(origin=np.array([t[0], t[1], t[2]]), normal = np.array([0, 0, 1]), size=(w,w), n = (2,2))
        pl.update()
        pl_space.append(pl)
        
    
    ax.scatter(t_space[0,:],t_space[1,:],t_space[2,:], color = 'b')
    objectPoints = pl.get_points()
    ax.scatter(objectPoints[0,:],objectPoints[1,:],objectPoints[2,:], color = 'g')
    #plt.show()
    
    #%%
    cam.set_t(pl.origin[0], pl.origin[1], pl.origin[2])
    
    
    
    #%%
    
    for plane in pl_space:
        cam.set_t(plane.origin[0], plane.origin[1], plane.origin[2])
        ippe_tvec_error, ippe_rmat_error, pnp_tvec_error, pnp_rmat_error = run_single(cam, plane, plot = True)
        
    
    
    #%%
    #Project points in camera
    imagePoints = np.array(cam.project(objectPoints,quant_error=False))
    
    #Add Gaussian noise in pixel coordinates
    #imagePoints = addnoise_imagePoints(imagePoints, mean = 0, sd = 1)
    
    #Show projected points
    
    cam.plot_image(imagePoints, pl.get_color())
    
        
        
        
    
    #%%
    


def run_point_distribution_test(cam, objectPoints):
    #Project points in camera
    imagePoints = np.array(cam.project(objectPoints,quant_error=False))
    
    #Add Gaussian noise in pixel coordinates
    imagePoints = addnoise_imagePoints(imagePoints, mean = 0, sd = 1)
    
    #Show porjected points
    cam.plot_image(imagePoints, pl.get_color())
    
    #Show the effect of the homography normalization    
    show_homo2d_normalization(imagePoints)    
    
    #Calculate the pose using solvepnp and plot the image points
    pnp_tvec, pnp_rmat = pose_pnp(objectPoints, imagePoints, cam.K)    
    pnpCam = cam.clone_withPose(pnp_tvec, pnp_rmat)
    pnpCam.plot_image(imagePoints, pl.get_color())
    
    #Calculate the pose using IPPE and plot the image points
    normalizedimagePoints = cam.get_normalized_pixel_coordinates(imagePoints)
    ippe_tvec, ippe_rmat = pose_ippe_best(objectPoints, normalizedimagePoints)    
    ippeCam = cam.clone_withPose(ippe_tvec, ippe_rmat)    
    ippeCam.plot_image(imagePoints, pl.get_color())
    
    
    ippe_tvec1,ippe_rmat1,ippe_tvec2,ippe_rmat2 = pose_ippe_both(objectPoints, normalizedimagePoints)
    ippeCam1 = cam.clone_withPose(ippe_tvec1, ippe_rmat1)    
    ippeCam2 = cam.clone_withPose(ippe_tvec2, ippe_rmat2)   
    ippeCam2.plot_image(imagePoints, pl.get_color())
    
    
    
    #Calculate errors
    pnp_tvec_error, pnp_rmat_error = calc_estimated_pose_error(cam.get_tvec(), cam.R, pnpCam.get_tvec(), pnp_rmat)
    ippe_tvec_error, ippe_rmat_error = calc_estimated_pose_error(cam.get_tvec(), cam.R, ippeCam.get_tvec(), ippe_rmat)
    
    # Print errors
    
    # IPPE pose estimation errors
    print ("----------------------------------------------------")
    print ("Translation Errors")
    print("IPPE    : %f" % ippe_tvec_error)
    print("solvePnP: %f" % pnp_tvec_error)
    
    # solvePnP pose estimation errors
    print ("----------------------------------------------------")
    print ("Rotation Errors")
    print("IPPE    : %f" % ippe_rmat_error)
    print("solvePnP: %f" % pnp_rmat_error)
    
    cams = [cam, ippeCam1, ippeCam2, pnpCam]
    planes = [pl]
    plot3D(cams, planes)
    

if __name__ == '__main__':
    #Init camera an plane
    cam,pl = init()
    objectPoints = pl.get_points()
    #create_uniform_cam_poses()
    run_point_distribution_test(cam, objectPoints)
    
    
    