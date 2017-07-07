# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.linalg import expm, rq, det, inv
import matplotlib.pyplot as plt
import numpy as np
from math import atan
from vision.rt_matrix import rotation_matrix


class Camera(object):
    """ Class for representing pin-hole cameras. """
    def __init__(self):
        """ Initialize P = K[R|t] camera model. """
        self.P = np.eye(3,4)
        self.K = np.eye(3) # calibration matrix
        self.R = np.eye(4) # rotation
        self.t = np.eye(4) # translation
        self.Rt = np.eye(4)
        self.fx = 1.
        self.fy = 1.
        self.cx = 0.
        self.cy = 0.
        self.img_width = 1280
        self.img_height = 960  
    
    def clone_withPose(self, tvec, rmat):
        new_cam = Camera()
        new_cam.K = self.K
        new_cam.set_R_mat(rmat)
        new_cam.set_t(tvec[0], tvec[1],  tvec[2])
        new_cam.set_P()
        new_cam.img_height = self.img_height
        new_cam.img_width = self.img_width
        return new_cam
        
    
    def set_P(self):
        # P = K[R|t]
        # P is a 3x4 Projection Matrix (from 3d euclidean to image)
        #self.Rt = hstack((self.R, self.t))  
        self.update_Rt()
        self.P = np.dot(self.K, self.Rt[:3,:4])
    
    def set_K(self, fx, fy, cx,cy):
        # K is the 3x3 Camera matrix
        # fx, fy are focal lenghts expressed in pixel units
        # cx, cy is a principal point usually at image center
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.mat([[fx, 0, cx], 
                      [0,fy,cy], 
                      [0,0,1]])
    def update_Rt(self):
        self.Rt = np.dot(self.t,self.R)
    
    def set_R_axisAngle(self,x,y,z, alpha):
        """  Creates a 3D [R|t] matrix for rotation
        around the axis of the vector defined by (x,y,z) 
        and an alpha angle.""" 
        #Normalize the rotation axis a
        a = np.array([x,y,z])
        a = a / np.linalg.norm(a)
        
        #Build the skew symetric
        a_skew = np.mat([[0,-a[2],a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(4)    
        R[:3,:3] = expm(a_skew*alpha)
        self.R = R
        self.update_Rt()
    
    def set_R_mat(self,R):
        self.R = R
        self.update_Rt()    
        
    
    def set_t(self, x,y,z):
        #self.t = array([[x],[y],[z]])
        self.t = np.eye(4)
        self.t[:3,3] = np.array([x,y,z]) 
        self.update_Rt()
    
    def set_world_position(self, x,y,z):
        cam_world = np.array([-x,-y,-z,1]).T
        t = np.dot(self.R,cam_world)
        self.set_t(t[0], t[1],  t[2])
        self.update_Rt()
    
    def get_normalized_pixel_coordinates(self, X):
        """
        These are in normalised pixel coordinates. That is, the effects of the 
        camera's intrinsic matrix and lens distortion are corrected, so that 
        the Q projects with a perfect pinhole model.
        """
        return np.dot(inv(self.K), X)
    
    
    def get_tvec(self):
        tvec = self.t[:,3]
        return tvec
        
        
        
    def get_world_position(self):
        t = np.dot(inv(self.Rt), np.array([0,0,0,1]))
        return t
        
        
    def project(self,X, quant_error=False):
        """  Project points in X (4*n array) and normalize coordinates. """
        self.set_P()
        x = np.dot(self.P,X)
        for i in range(x.shape[1]):              
          x[:,i] /= x[2,i]
        if(quant_error):
            x = np.round(x, decimals=0)            
        return x
    
    def plot_image(self, imgpoints, points_color):
        #%% show Image
        # plot projection
        plt.figure()
        plt.plot(imgpoints[0],imgpoints[1],'.',color = points_color)
        #we add a key point to help us see orientation of the points
        plt.plot(imgpoints[0,0],imgpoints[1,0],'.',color = 'blue')
        plt.xlim(0,self.img_width)
        plt.ylim(0,self.img_height)
        plt.gca().invert_yaxis()
        plt.show()
        
        
    def factor(self):
        """  Factorize the camera matrix into K,R,t as P = K[R|t]. """
        # factor first 3*3 part
        K,R = rq(self.P[:,:3])
        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if det(T) < 0:
            T[1,1] *= -1
        self.K = np.dot(K,T)
        self.R = np.dot(T,R) # T is its own inverse
        self.t = np.dot(inv(self.K),self.P[:,3])
        return self.K, self.R, self.t
    
    def fov(self):
        """ Calculate field of view angles (grads) from camera matrix """
        fovx = np.rad2deg(2 * atan(self.img_width / (2. * self.fx)))
        fovy = np.rad2deg(2 * atan(self.img_height / (2. * self.fy)))
        return fovx, fovy
    
    def move(self, x,y,z):
        Rt = np.identity(4);
        Rt[:3,3] = np.array([x,y,z])
        self.P = np.dot(self.K, self.Rt)
    
    def rotate(self, axis, angle):
        """ rotate camera around a given axis in world coordinates"""        
        R = rotation_matrix(axis, angle)
        self.Rt = np.dot(R, self.Rt)
        self.R[:3,:3] = self.Rt[:3,:3]
        self.t[:3,3] = self.Rt[:3,3]
    
    def rotate_x(self,angle):
        self.rotate(np.array([1,0,0],dtype=np.float32), angle)
    
    def rotate_y(self,angle):
        self.rotate(np.array([0,1,0],dtype=np.float32), angle)
    
    def rotate_z(self,angle):
        self.rotate(np.array([0,0,1],dtype=np.float32), angle)
        
        
        
        



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
