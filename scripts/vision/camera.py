# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy import *
from scipy.linalg import expm, sinm, cosm, rq, det, inv
import matplotlib.pyplot as plt
import numpy as np
from math import atan
from vision.rt_matrix import *


class Camera(object):
    """ Class for representing pin-hole cameras. """
    def __init__(self):
        """ Initialize P = K[R|t] camera model. """
        self.P = eye(3,4)
        self.K = eye(3) # calibration matrix
        self.R = eye(4) # rotation
        self.t = eye(4) # translation
        self.Rt = eye(4)
        self.fx = 1.
        self.fy = 1.
        self.cx = 0.
        self.cy = 0.
        self.img_width = 1280
        self.img_height = 960      
        
    
    def set_P(self):
        # P = K[R|t]
        # P is a 3x4 Projection Matrix (from 3d euclidean to image)
        #self.Rt = hstack((self.R, self.t))        
        self.P = dot(self.K, self.Rt[:3,:4])
    
    def set_K(self, fx, fy, cx,cy):
        # K is the 3x3 Camera matrix
        # fx, fy are focal lenghts expressed in pixel units
        # cx, cy is a principal point usually at image center
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = mat([[fx, 0, cx], 
                      [0,fy,cy], 
                      [0,0,1]])
    
    def set_R(self,x,y,z, alpha):
        """  Creates a 3D [R|t] matrix for rotation
        around the axis of the vector defined by (x,y,z) 
        and an alpha angle.""" 
        #Normalize the rotation axis a
        a = array([x,y,z])
        a = a / np.linalg.norm(a)
        
        #Build the skew symetric
        a_skew = mat([[0,-a[2],a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = eye(4)    
        R[:3,:3] = expm(a_skew*alpha)
        self.R = R
        self.Rt = dot(self.t,self.R)
    
    def set_t(self, x,y,z):
        #self.t = array([[x],[y],[z]])
        self.t = eye(4)
        self.t[:3,3] = array([x,y,z])     
        self.Rt = dot(self.t,self.R)
    
    def set_world_position(self, x,y,z):
        cam_world = transpose(array([-x,-y,-z,1]))
        t = dot(self.R,cam_world)
        self.set_t(t[0], t[1],  t[2])
        self.Rt = dot(self.t,self.R)
    
    def get_world_position(self):
        t = dot(inv(self.Rt), array([0,0,0,1]))
        return t
        
        
    def project(self,X):
        """  Project points in X (4*n array) and normalize coordinates. """
        x = dot(self.P,X)
        for i in range(3):
          x[i] /= x[2]
        return x
        
        
    def factor(self):
        """  Factorize the camera matrix into K,R,t as P = K[R|t]. """
        # factor first 3*3 part
        K,R = rq(self.P[:,:3])
        # make diagonal of K positive
        T = diag(sign(diag(K)))
        if det(T) < 0:
            T[1,1] *= -1
        self.K = dot(K,T)
        self.R = dot(T,R) # T is its own inverse
        self.t = dot(inv(self.K),self.P[:,3])
        return self.K, self.R, self.t
    
    def fov(self):
        """ Calculate field of view angles (grads) from camera matrix """
        fovx = rad2deg(2 * atan(self.img_width / (2. * self.fx)))
        fovy = rad2deg(2 * atan(self.img_height / (2. * self.fy)))
        return fovx, fovy
    
    def move(self, x,y,z):
        Rt = identity(4);
        Rt[:3,3] = array([x,y,z])
        self.P = dot(self.K, self.Rt)
    
    def rotate(self, axis, angle):
        """ rotate camera around a given axis in world coordinates"""        
        R = rotation_matrix(axis, angle)
        self.Rt = dot(R, self.Rt)
        self.R[:3,:3] = self.Rt[:3,:3]
        self.t[:3,3] = self.Rt[:3,3]
    
    def rotate_x(self,angle):
        self.rotate(array([1,0,0],dtype=np.float32), angle)
    
    def rotate_y(self,angle):
        self.rotate(array([0,1,0],dtype=np.float32), angle)
    
    def rotate_z(self,angle):
        self.rotate(array([0,0,1],dtype=np.float32), angle)
        
        
        
        



#cam = Camera()

##Test that projection matrix doesnt change rotation and translation
#
#cam.set_world_position(0,0,-2.5)
#R1= cam.R
#t1 = cam.t
#Rt1 = cam.Rt
#pos1 = cam.get_world_position()
#cam.set_P()
#R2 = cam.R
#t2 = cam.t
#Rt2 = cam.Rt
#pos2 = cam.get_world_position()
#print pos1-pos2
#print R1 - R2
#print t1 - t2
#print Rt1 - Rt2
#
#
#print "------------------------------"
##Test that rotate function doesnt change translation matrix
#
#cam.set_world_position(0,0,-2.5)
#R1= cam.R
#t1 = cam.t
#Rt1 = cam.Rt
#pos1 = cam.get_world_position()
#cam.set_P()
#
#cam.rotate_y(deg2rad(+20.))
#cam.rotate_y(deg2rad(+20.))
#cam.set_P()
#R2 = cam.R
#t2 = cam.t
#Rt2 = cam.Rt
#pos2 = cam.get_world_position()
#print pos1-pos2
#print R1 - R2
#print t1 - t2
#print Rt1 - Rt2
