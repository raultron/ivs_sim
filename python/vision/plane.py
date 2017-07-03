#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:04:45 2017

@author: lracuna
"""
import numpy as np
from vision.rt_matrix import *

class Plane(object):
    """ Class for representing a 3D grid plane based on a point and a normal."""
    def __init__(self, origin=np.array([0, 0, 0]), normal = np.array([0, 0, 1]), grid_size=(3,3), grid_step = 1):
        self.origin = origin
        self.normal = normal
        self.grid_size = grid_size
        self.grid_step = grid_step   
        self.plane_points = None
        self.color = (1,0,0)
        self.angle = 0.
        self.R = np.eye(4)

    
    def update(self):
        #we create a plane in the x-y plane
        
        # create x,y
        x_range = range(int(round(self.grid_size[0]/self.grid_step)))
        y_range = range(int(round(self.grid_size[1]/self.grid_step)))
        xx, yy = np.meshgrid(x_range, y_range)
        # center the plane
        xx = (xx.astype(np.float32))*self.grid_step - (x_range[-1]*self.grid_step/2.)
        yy = (yy.astype(np.float32))*self.grid_step - (y_range[-1]*self.grid_step/2.)
        
        # calculate corresponding z
        hh = np.ones_like(xx, dtype=np.float32)
        zz = np.zeros_like(xx, dtype=np.float32)
        
        self.plane_points = np.array([xx.ravel(),yy.ravel(),zz.ravel(), hh.ravel()], dtype=np.float32) 
        
        self.plane_points_basis = self.plane_points
        
#        #we rotate the plane around the normal axis by the given angle
#        
#        if self.angle!=0.:
#            self.R = rotation_matrix(self.normal, self.angle)
#            self.plane_points = dot(self.R, self.plane_points)
#        
#        #we now align the plane to the required normal
#        
#        current_normal = array([1,0,0])
#        desired_normal = self.normal
#        if not (current_normal == desired_normal).all():
#            self.R = R = rotation_matrix_from_two_vectors(current_normal,desired_normal)    
#            self.plane_points = dot(self.R, self.plane_points)
#        
#       
##        
        # translate        
        self.plane_points[0] += self.origin[0]
        self.plane_points[1] += self.origin[1]
        self.plane_points[2] += self.origin[2]
        
        
        
        self.xx = xx
        self.yy = yy
        self.zz = zz
         
                
    
    def get_points(self):
        return self.plane_points
    
    def get_points_basis(self):
        return self.plane_points_basis
    
    def get_mesh(self):
        return self.xx, self.yy, self.zz
    
    def get_color(self):
        return self.color
    
    def set_origin(self, origin):
        self.origin = origin
    
    def set_normal(self, normal):
        self.normal = normal
    
    def set_grid(self, grid_x, grid_y, grid_step):
        self.grid_size = (grid_x, grid_y)
        self.grid_step = grid_step
    
    def set_color(self,color):
        self.color = color
        
    def rotate(self, axis, angle):
        """ rotate plane points around a given axis in world coordinates"""
        self.plane_points[0] -= self.origin[0]
        self.plane_points[1] -= self.origin[1]
        self.plane_points[2] -= self.origin[2]
        
        R = rotation_matrix(axis, angle)
        self.plane_points = np.dot(R, self.plane_points)
        
        # return translation
        self.plane_points[0] += self.origin[0]
        self.plane_points[1] += self.origin[1]
        self.plane_points[2] += self.origin[2]
    
    def rotate_x(self,angle):
        self.rotate(np.array([1,0,0],dtype=np.float32), angle)
    
    def rotate_y(self,angle):
        self.rotate(np.array([0,1,0],dtype=np.float32), angle)
    
    def rotate_z(self,angle):
        self.rotate(np.array([0,0,1],dtype=np.float32), angle)
        
        
        
    
        
    
        
    
        
        
        
        