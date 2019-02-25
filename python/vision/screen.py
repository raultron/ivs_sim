#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:04:45 2017

@author: lracuna
"""
from vision.plane import Plane
import numpy as np


class Screen(Plane):
    """ Class for representing a 3D LCD screen"""
    def __init__(self, width=640, height=480, pixel_pitch=0.270,
                 curvature_radius=2.0):
        self.set_pixel_pitch(pixel_pitch)
        self.set_resolution(width, height)
        self.set_grid()
        # Curvature radius in meters, a value of 0.0 means a plane
        self.curvature_radius = curvature_radius
        Plane.__init__(self)

    def set_resolution(self, width, height):
        """ Set the resolution in pixels """
        self.width = width
        self.height = height
        self.resolution = (width, height)
        # self.aspect_ratio = TODO EQUATION

    def set_pixel_pitch(self, pixel_pitch):
        """ pixel_pitch: float
                         In milimiters """
        self.pixel_pitch = pixel_pitch
        self.pixel_pitch_cm = pixel_pitch / 10.  # In centimeters
        self.pixel_pitch_m = self.pixel_pitch_cm / 100.  # In meters

    def set_grid(self):
        self.grid_size = (self.width*self.pixel_pitch_m,
                          self.height*self.pixel_pitch_m)
        self.grid_step = self.pixel_pitch_m

    def update(self):
        if self.curvature_radius == 0:
            super(Screen, self).update()
        else:
            self.update_curved()
            print("curved screen")

    def update_curved(self):
        # We create a plane in the x-y plane
        # Create x,y meshgrid
        x_range = range(int(round(self.grid_size[0]/self.grid_step)))
        y_range = range(int(round(self.grid_size[1]/self.grid_step)))
        xx, yy = np.meshgrid(x_range, y_range)

        # Center the plane
        xx = (xx.astype(np.float32))*self.grid_step - (x_range[-1]*self.grid_step/2.)
        yy = (yy.astype(np.float32))*self.grid_step - (y_range[-1]*self.grid_step/2.)

        # calculate corresponding z
        teta = np.arccos((xx)/(self.curvature_radius/2.0))
        zz = self.curvature_radius - self.curvature_radius*np.sin(teta)

        hh = np.ones_like(xx, dtype=np.float32)
        self.plane_points = np.array([xx.ravel(), yy.ravel(), zz.ravel(),
                                      hh.ravel()], dtype=np.float32)
        self.plane_points_basis = self.plane_points

        # translate
        self.plane_points[0] += self.origin[0]
        self.plane_points[1] += self.origin[1]
        self.plane_points[2] += self.origin[2]

        self.xx = xx
        self.yy = yy
        self.zz = zz


if __name__ == "__main__":
    from vision.camera import Camera
    from vision.plot_tools import plot3D

    cam = Camera()
    cam.set_K(fx=100, fy=100, cx=640, cy=480)
    cam.set_width_heigth(1280, 960)

    """ Initial camera pose looking stratight down into the plane model """
    cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180))
    cam.set_t(0.0, 0.0, 1.5, frame='world')

    """ Screen for the control points """
    # 10x10 grid on a squared screen of 1920x1920 with a pixel pitch of 0.270
    # I have to change this function a little bit so it is easier to simulate

    sc = Screen(width=10,height=10, pixel_pitch= 192*0.270)
    sc.update()
    fiducial_points = sc.get_points() # If you want to check the points
    """ Plot camera axis and plane """
    cams = [cam]
    planes = [sc]
    plot3D(cams, planes)

    """ Show image """
    sc.plot_points()