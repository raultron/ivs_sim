#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:24:01 2018

@author: lracuna
"""
from vision.circular_plane import CircularPlane
from optimize.utils import flatten_points, unflatten_points
from vision.camera import Camera
from optimize import objectives as obj
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import autograd.numpy as np
from autograd import grad
from autograd.optimizers import adam

class OptimalPointsSim(object):
  """ Class that defines and optimization to obtain optimal control points
  configurations for homography and plannar pose estimation. """

  def __init__(self):
    """ Definition of a simulated camera """
    self.cam = Camera()
    self.cam.set_K(fx = 100,fy = 100,cx = 640,cy = 480)
    self.cam.set_width_heigth(1280,960)

    """ Initial camera pose looking stratight down into the plane model """
    self.cam.set_R_axisAngle(1.0,  0.0,  0.0, np.deg2rad(180))
    self.cam.set_t(0.0,0.0,1.5, frame='world')

    """ Plane for the control points """
    self.pl = CircularPlane(radius=1.5)
    self.pl.random(n =200, r = 0.01, min_sep = 0.001)

  def run(self):
    self.objectPoints = self.pl.get_points()
    self.init_params = flatten_points(self.objectPoints, type='object_plane')

    self.objective1 = lambda params: obj.matrix_condition_number_autograd(params, self.cam.P, normalize = False)
    self.objective2 = lambda params,iter: obj.matrix_condition_number_autograd(params, self.cam.P, normalize = True)

    print("Optimizing condition number...")
    #out =  minimize(value_and_grad(objective2), init_params, jac=True,
    #                      method='COBYLA')
    #print out
    #optimized_params = out['x']
    objective_grad = grad(self.objective2)
    self.optimized_params = adam(objective_grad, self.init_params, step_size=0.11,
                                num_iters=3000, callback = self.plot_points)

  def plot_points(self,params, iter, gradient):
    global cn, s1, s2, c1
    x = params[::2]
    y = params[1::2]
    object_points = unflatten_points(params, type = 'object_plane')
    image_points = np.dot(self.cam.P,object_points)
    image_pts = image_points/image_points[2,:]
    img_flat = flatten_points(image_points, type = 'image')
    if iter == 0:
      self.pw_w = pg.PlotWidget()
      self.pw_i = pg.PlotWidget()
      self.pw_cn =  pg.PlotWidget()
      self.pw_w.setAspectLocked(lock=True, ratio=1)
      self.pw_i.setAspectLocked(lock=True, ratio=1)
      self.s1 = pg.ScatterPlotItem()
      self.s2 = pg.ScatterPlotItem()
      self.c1 = pg.PlotCurveItem()
      self.pw_w.show()
      self.pw_i.show()
      self. pw_cn.show()
      self.cn = []
      self.cn.append(self.objective1(params))
      self.c1.setData(self.cn)
      self.s1 = pg.ScatterPlotItem(x,y)
      self.s2 = pg.ScatterPlotItem(img_flat[::2],img_flat[1::2])
      self.pw_w.addItem(self.s1)
      self.pw_i.addItem(self.s2)
      self.pw_cn.addItem(self.c1)
    else:
      self.cn.append(self.objective1(params))
      self.s1.setData(x,y)
      self.s2.setData(img_flat[::2],img_flat[1::2])
      self.c1.setData(self.cn)
    QtGui.QApplication.processEvents()


sim = OptimalPointsSim()
sim.run()