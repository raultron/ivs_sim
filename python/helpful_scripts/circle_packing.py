#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:41:09 2017

@author: lracuna
"""

#!/usr/bin/env python

"""
This program uses a simple implementation of the ADMM algorithm to solve 
the circle packing problem.
We solve
    minimize 1
    subject to |x_i - x_j| > 2R,
               R < x_i, y_i < L - R
We put a bunch of equal radius balls inside a square.
Type --help to see the options of the program.
Must create a directory .figs.
Guilherme Franca
guifranca@gmail.com
November 2015
"""

import sys, os, optparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def nonoverlap(a, i, omega, R):
    """No overlap constraint.
    This function receives a 1D array which is the row of a matrix.
    Each element is a vector. i is which row we are passing.
    """
    nonzeroi = np.nonzero(omega[i])[0]
    x = a
    n1, n2 = a[nonzeroi]
    vec = n1 - n2
    norm = np.linalg.norm(vec)
    if norm < 2*R: # push the balls appart
        disp = R - norm/2
        x[nonzeroi] = n1 + (disp/norm)*vec, n2 - (disp/norm)*vec
    return x
    
def insidebox(a, i, omega, R, L):
    """Keep the balls inside the box."""
    j = np.nonzero(omega[i])[0][0]
    x = a
    n = a[j]
    if n[0] < R:
        x[j,0] = R
    elif n[0] > L-R:
        x[j,0] = L-R
    if n[1] < R:
        x[j,1] = R
    elif n[1] > L-R:
        x[j,1] = L-R
    return x

def make_graph(t, z, imgpath, R, L):
    """Create a plot of a given time.
    z contains a list of vectors with the position of the center of
    each ball. t is the iteration time.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle('t=%i' % t)
    ax.add_patch(Rectangle((0,0), L, L, fill=False,
                 linestyle='solid', linewidth=2, color='blue'))
    plt.xlim(-0.5, L+0.5)
    plt.ylim(-0.5, L+0.5)
    plt.axes().set_aspect('equal')
    colors = iter(plt.cm.prism_r(np.linspace(0,1,N)))
    for x in z:
        c = next(colors)
        ax.add_patch(Circle(x, radius=R, color=c, alpha=.6))
    plt.axis('off')
    fig.tight_layout()
    fig.savefig(imgpath % t, format='png')
    print imgpath
    plt.close(fig)

    
def make_omega(N):
    """Topology matrix
    Columns label variables, and rows the functions.
    You must order all the "nonoverlap" functions first
    and the "inside box" function last.
    We also create a vectorized version of omega.
    """
    o1 = []
    o2 = []
    one = np.array([1,1])
    zero = np.array([0,0])
    # TODO: this is the most expensive way of creating these matrices.
    # Maybe improve this.
    for i in range(N):
        for j in range(i+1, N):
            row1 = [0]*N
            row1[i], row1[j] = 1, 1
            o1.append(row1)
            row2 = [zero]*N
            row2[i], row2[j] = one, one
            o2.append(row2)
    for i in range(N):
        row = [0]*N
        row[i] = 1
        o1.append(row)
        row2 = [zero]*N
        row2[i] = one
        o2.append(row2)
    
    o1 = np.array(o1)
    o2 = np.array(o2)
    return o1, o2


###############################################################################
if __name__ == '__main__': 
    usg = "%prog -L box -R radius -N balls -M iter [-r rate -o output]"
    dsc = "Use ADMM optimization algorithm to fit balls into a box."
    parser = optparse.OptionParser(usage=usg, description=dsc)
    parser.add_option('-L', '--box_size', action='store', dest='L',
                        type='float', help='size of the box')
    parser.add_option('-R', '--radius', action='store', dest='R',
                        type='float', help='radius of the balls')
    parser.add_option('-N', '--num_balls', action='store', dest='N',
                        type='int', help='number of balls')
    parser.add_option('-M', '--iter', action='store', dest='M',
                        type='int', help='number of iterations')
    parser.add_option('-r', '--rate', action='store', dest='rate',
                    default=10, type='float', help='frame rate for the movie')
    parser.add_option('-o', '--output', action='store', dest='out',
                    default='out.mp4', type='str', help='movie output file')
    parser.add_option('-a', '--alpha', action='store', dest='alpha',
                    default=0.05, type='float', help='alpha parameter')
    parser.add_option('-p', '--rho', action='store', dest='rho',
                    default=0.5, type='float', help='rho parameter')
    options, args = parser.parse_args()
    if not options.L:
        parser.error("-L option is mandatory")
    if not options.R:
        parser.error("-R option is mandatory")
    if not options.N:
        parser.error("-N option is mandatory")
    if not options.M:
        parser.error("-M option is mandatory")

    # initialization
    L = options.L
    R = options.R
    N = options.N
    max_iter = options.M
    rate = options.rate
    output = options.out
    
    omega, omega_vec = make_omega(N)
    num_funcs = len(omega)
    num_vars = len(omega[0])
    s = (num_funcs, num_vars, 2) 
    alpha = float(options.alpha)
    x = np.ones(s)*omega_vec
    z = np.random.random_sample(size=(num_vars, 2))+\
        (L/2.)*np.ones((num_vars, 2))
    zz = np.array([z]*num_funcs)*omega_vec
    u = np.ones(s)*omega_vec
    n = np.ones(s)*omega_vec
    rho = float(options.rho)*omega_vec

    # performing optimization
    if not os.path.exists('.figs'):
        os.makedirs('.figs')
    os.system("rm -rf .figs/*")
    imgpath = '.figs/fig%04d.png' 
    
    for k in range(max_iter):
        n = zz - u
        # proximal operator
        for i in range(num_funcs):
            if i < num_funcs - num_vars:
                x[i] = nonoverlap(n[i], i, omega, R)
            else:
                x[i] = insidebox(n[i], i, omega, R, L)
        m = x + u
        z = np.sum(rho*m, axis=0)/np.sum(rho, axis=0)
        zz = np.array([z]*num_funcs)*omega_vec
        u = u + alpha*(x-zz)

        if k == (max_iter-1):
            make_graph(k, z, imgpath, R, L)
        print "doing %i/%i" % (k, max_iter)
    
    print "Generating animation '%s' ..." % (output)
    os.system("ffmpeg -y -r %f -sameq -i %s %s > /dev/null 2>&1" % \
                                            (rate, imgpath, output))
    #os.system("rm -rf .figs/*")
    #os.rmdir('.figs')
    print "Done!"
    print "Playing ..."
    os.system("mplayer %s > /dev/null 2>&1" % output)