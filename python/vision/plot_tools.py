# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:38:35 2017

@author: racuna
"""
import numpy as np
from mayavi import mlab


def display_figure():
    """Display the Mayavi Opengl figure"""
    mlab.show()


def plot3D_cam(cam, axis_scale=0.2):
    """Plots the camera axis in a given position and orientation in 3D space

    Parameters
    ----------
    cam : :obj:`Camera`
            Object of the type Camera, with a proper Rt matrix.
    axis_scale : int, optional
            The Scale of the axis in 3D space.

    Returns
    -------
    None
    """
    # Coordinate Frame of camera
    cam_axis_x = np.array([1, 0, 0, 1]).T
    cam_axis_y = np.array([0, 1, 0, 1]).T
    cam_axis_z = np.array([0, 0, 1, 1]).T

    cam_axis_x = np.dot(cam.R.T, cam_axis_x)
    cam_axis_y = np.dot(cam.R.T, cam_axis_y)
    cam_axis_z = np.dot(cam.R.T, cam_axis_z)

    cam_world = cam.get_world_position()

    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2],
                  cam_axis_x[0], cam_axis_x[1], cam_axis_x[2],
                  line_width=3, scale_factor=axis_scale,
                  color=(1-axis_scale, 0, 0))

    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2],
                  cam_axis_y[0], cam_axis_y[1], cam_axis_y[2],
                  line_width=3, scale_factor=axis_scale,
                  color=(0, 1-axis_scale, 0))

    mlab.quiver3d(cam_world[0], cam_world[1], cam_world[2],
                  cam_axis_z[0], cam_axis_z[1], cam_axis_z[2],
                  line_width=3, scale_factor=axis_scale,
                  color=(0, 0, 1-axis_scale))


def plotPoints3D(fiducial_space, scale=0.01):
    """Plots a set of points from a fiducial space on 3D Space

    Parameters
    ----------
    fiducial_space : object
            This can be a Plane() object or similar fiducial space objects
    scale : float, optional
            Scale of each one of the plotted points

    Returns
    -------
    None
    """

    fiducial_points = fiducial_space.get_points()

    mlab.points3d(fiducial_points[0], fiducial_points[1], fiducial_points[2],
                  scale_factor=scale, color=fiducial_space.get_color())


def plot3D(cams, planes):
    """Plots a set of cameras and a set of fiducial planes on the 3D Space

    Parameters
    ----------
    cams : list
            List of objects of the type Camera each one with a proper Rt matrix
    planes : list
            List of objects of the type Plane

    Returns
    -------
    None
    """
    # mlab.figure(figure=None, bgcolor=(0.1,0.5,0.5), fgcolor=None,
    #              engine=None, size=(400, 350))
    axis_scale = 0.05
    for cam in cams:
        plot3D_cam(cam, axis_scale)
    for fiducial_space in planes:
        # Plot plane points in 3D
        plotPoints3D(fiducial_space)
    display_figure()
