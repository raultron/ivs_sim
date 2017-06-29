# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:16:21 2017

@author: lracuna
"""
import camera_calibration
import yaml
import numpy as np

def loadCalibrationFile(filename, cname):
    """ Load calibration data from a file.
    This function returns a `sensor_msgs/CameraInfo`_ message, based
    on the filename parameter.  An empty or non-existent file is *not*
    considered an error; a null CameraInfo being provided in that
    case.
    :param filename: location of CameraInfo to read
    :param cname: Camera name.
    :returns: `sensor_msgs/CameraInfo`_ message containing calibration,
              if file readable; null calibration message otherwise.
    :raises: :exc:`IOError` if an existing calibration file is unreadable.
    """
    ci = camera_calibration.Camera
    try:
        f = open(filename)
        calib = yaml.load(f)
        if calib is not None:
            if calib['camera_name'] != cname:
                print("[" + cname + "] does not match name " +
                              calib['camera_name'] + " in file " + filename)

            # fill in CameraInfo fields
            #ci.width = calib['image_width']
            #ci.height = calib['image_height']
            #ci.distortion_model = calib['distortion_model']
            #ci.D = calib['distortion_coefficients']['data']
            ci.K = calib['camera_matrix']['data']
            ci.R = calib['rectification_matrix']['data']
            ci.P = calib['projection_matrix']['data']

    except IOError:                     # OK if file did not exist
        pass

    return ci
    
calibs = list()
fx = list()
fy = list()
cx = list()
cy = list()
for i in range(1,18):
    filename = "logitech_camera_calibration/logitech_cam_dyn"+str(i)+".yaml"
    print (filename)
    ci = loadCalibrationFile(filename, "logitech_cam")
    fx.append(ci.K[0])
    fy.append(ci.K[4])
    cx.append(ci.K[2])
    cy.append(ci.K[5])

fx = np.array(fx)
fy = np.array(fy)
cx = np.array(cx)
cy = np.array(cy)

#%%
filename = "logitech_camera_calibration/logitech_cam_ground_truth_2.yaml"
print (filename)
ci = loadCalibrationFile(filename, "logitech_cam")

fx_gt = ci.K[0]
fy_gt = ci.K[4]
cx_gt = ci.K[2]
cy_gt = ci.K[5]

print (" fx factor: ")
print (fx / fx_gt)
print (" fy factor: ")
print (fy / fy_gt)
print (" fx factor / fy factor: ")
print ((fx / fx_gt) / (fy / fy_gt) )
print (" cx factor: ")
print (cx / cx_gt)
print (" cy factor: ")
print (cy / cy_gt)
print (" cx factor / cy factor: ")
print ((cx / cx_gt) / (cy / cy_gt))
print (" fx factor / cx factor: ")
print ((fx / fx_gt) / (cx / cx_gt))



