#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@Time    : 15.12.17 14:16
@File    : plane_distribution.py
@author: Yue Hu
"""

import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D
from vision.plane import Plane
import Rt_matrix_from_euler_t as Rt_matrix_from_euler_t


def uniform_sphere(theta_params=(0, 360, 10), phi_params=(0, 90, 10), r=1., plot=False):
    """n points distributed evenly on the surface of a unit sphere
    theta_params: tuple (min = 0,max = 360, N divisions = 10)
    phi_params: tuple (min =0,max =90, N divisions = 10)
    r: radius of the sphere
    n_theta: number of points in theta
    n_phi: number of points in phi

    """
    space_theta = np.linspace(np.deg2rad(theta_params[0]), np.deg2rad(theta_params[1]), theta_params[2])
    space_theta = space_theta[:-1].copy() # TODO delete 360 is same to 0
    space_phi = np.linspace(np.deg2rad(phi_params[0]), np.deg2rad(phi_params[1]), phi_params[2])
    theta, phi = np.meshgrid(space_theta, space_phi)
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        plt.show()

    return x, y, z


def plot3D_plane(plane, R=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), axis_scale=0.2):
    # Coordinate Frame of real camera
    # Camera axis
    pl_axis_x = np.array([1, 0, 0, 1]).T
    pl_axis_y = np.array([0, 1, 0, 1]).T
    pl_axis_z = np.array([0, 0, 1, 1]).T

    pl_axis_x = np.dot(R.T, pl_axis_x)
    pl_axis_y = np.dot(R.T, pl_axis_y)
    pl_axis_z = np.dot(R.T, pl_axis_z)

    pl_world = plane.origin

    mlab.quiver3d(pl_world[0], pl_world[1], pl_world[2], pl_axis_x[0], pl_axis_x[1], pl_axis_x[2], line_width=3,
                  scale_factor=axis_scale, color=(1 - axis_scale, 0, 0))
    mlab.quiver3d(pl_world[0], pl_world[1], pl_world[2], pl_axis_y[0], pl_axis_y[1], pl_axis_y[2], line_width=3,
                  scale_factor=axis_scale, color=(0, 1 - axis_scale, 0))
    mlab.quiver3d(pl_world[0], pl_world[1], pl_world[2], pl_axis_z[0], pl_axis_z[1], pl_axis_z[2], line_width=3,
                  scale_factor=axis_scale, color=(0, 0, 1 - axis_scale))


def plot3D(planes, height):
    for plane, h in zip(planes, height):
        # Plot plane points in 3D
        plane_points = plane.get_points()
        plot3D_plane(plane)
        mlab.points3d(plane_points[0], plane_points[1], plane_points[2], scale_factor=0.05, color=plane.get_color())
        mlab.points3d(plane_points[0, 0], plane_points[1, 0], plane_points[2, 0], scale_factor=0.05, color=(0., 0., 1.))
        # plane_corners2d = plane.get_corners()
        # plane_corners3d = np.eye(3,5)
        # plane_corners3d[0:2,0:4] = plane_corners2d[0:2,0:4]
        # plane_corners3d[2] =h
        # plane_corners3d[:,4]=plane_corners3d[:,0]
        # mlab.plot3d(plane_corners3d[0],plane_corners3d[1],plane_corners3d[2],color=(1,0,0))
    mlab.show()


def create_plane_distribution(plane_size=(0.3, 0.3), theta_params=(0, 360, 5), phi_params=(0, 70, 3),
                              r_params=(0.5, 1.0, 2), plot=False):
    # We extend the size of this plane to account for the deviation from a uniform pattern
    # plane.size = (plane.size[0] + deviation, plane.size[1] + deviation)
    planes = []
    d_space = np.linspace(r_params[0], r_params[1], r_params[2])
    t_list = []
    for d in d_space:
        xx, yy, zz = uniform_sphere(theta_params, phi_params, d, False)
        sphere_points = np.array([xx.ravel(), yy.ravel(), zz.ravel()], dtype=np.float32)
        t_list.append(sphere_points)
    t_space = np.hstack(t_list)
    # print t_space.shape
    for t in t_space.T:
        # We set one static plane at (0,0,0) in the world coordinate, this static plane is fixed!!!
        # We set a static camera straight up of this static plane, this camera is also fixed!!!
        # The relationship between static plane, static camera and new plane is : T = T1 * T2

        # we create a default plane with 4 points with a side lenght of w (meters)
        # The origin of each new plane is the point of sphere at each position
        real_origin = t
        plane = Plane(origin=real_origin, normal=np.array([0, 0, 1]), size=plane_size, n=(2, 2))
        plane.uniform()  # TODO
        planes.append(plane)

    # print len(planes)

    if plot:
        height = t_space[2]
        plot3D(planes, height)

    return planes


# ---------------------------------------------
create_plane_distribution(plot=True)
