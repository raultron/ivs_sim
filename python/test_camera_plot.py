#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:24:01 2018

@author: lracuna
"""
import numpy as np
from vision.circular_plane import CircularPlane
from vision.screen import Screen
from vision.camera import Camera
from vision.plot_tools import plot3D

cam = Camera()
cam.set_K(fx=100, fy=100, cx=640, cy=480)
cam.set_width_heigth(1280, 960)

""" Initial camera pose looking stratight down into the plane model """
cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180))
cam.set_t(0.0, 0.0, 1.5, frame='world')

""" Plane for the control points """
pl = CircularPlane(radius=0.5)
pl.set_color((1, 1, 0))
pl.random(n=200, r=0.01, min_sep=0.001)

""" Plot camera axis and plane """
cams = [cam]
planes = [pl]
plot3D(cams, planes)
