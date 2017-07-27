#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:53:05 2017

@author: lracuna
"""
#%%
from pose_sim import *

camR = np.load('camR.npy')
camt = np.load('camt.npy')
objectPoints = np.load('objectPoints_test_ippe.npy')

cam = Camera()
cam.set_K(fx = 800., fy = 800., cx = 320., cy = 240.)  #Camera Matrix
cam.img_width = 320.*2.
cam.img_height = 240.*2.

cam.R = camR
cam.t = camt
cam.set_P()

imagePoints = np.array(cam.project(objectPoints,False))


normalizedimagePoints = cam.get_normalized_pixel_coordinates(imagePoints)

ippe_tvec1,ippe_rmat1,ippe_tvec2,ippe_rmat2 = pose_ippe_both(objectPoints, normalizedimagePoints, False)
ippeCam1 = cam.clone_withPose(ippe_tvec1, ippe_rmat1)
ippeCam2 = cam.clone_withPose(ippe_tvec2, ippe_rmat2)

#ippeCam = cam.clone_withPose(ippe_tvec, ippe_rmat)


ippe_tvec_error1, ippe_rmat_error1 = calc_estimated_pose_error(cam.get_tvec(), cam.R, ippeCam1.get_tvec(), ippe_rmat1)
ippe_tvec_error2, ippe_rmat_error2 = calc_estimated_pose_error(cam.get_tvec(), cam.R, ippeCam2.get_tvec(), ippe_rmat2)

#%%
pnp_tvec, pnp_rmat = pose_pnp(objectPoints, imagePoints, cam.K, False, cv2.SOLVEPNP_ITERATIVE,False)
pnpCam = cam.clone_withPose(pnp_tvec, pnp_rmat)
pnp_tvec_error, pnp_rmat_error = calc_estimated_pose_error(cam.get_tvec(), cam.R, pnpCam.get_tvec(), pnp_rmat)