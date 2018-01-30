#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:52:11 2017

@author: lracuna
"""
import sys
sys.path.append("../")
sys.path.append("../vision/")
sys.path.append("../gdescent/")

import autograd.numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from vision.camera import Camera
from vision.plane import Plane
from vision.circular_plane import CircularPlane
from matplotlib import rc

mpl.pyplot.close("all")
mpl.rc('font',**{'family':'serif','serif':['Times']})
#mpl.rc('text', usetex=True)


Din = pickle.load( open( "icra_sim1_frontoparallel.p", "rb" ) )
cam = Din["Camera"]
pl = Din['Plane']



fig_width_pt = 245.71811  # Get this from LaTeX using \showthe\columnwidth

inch_per_cent = 0.393701
inches_per_pt = 1.0/72.27               # Convert pt to inch
#golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = 10*inch_per_cent      # height in inches
#fig_size =  [fig_width,fig_height]
params = {'backend': 'PDF',
          'axes.labelsize': 8,
          'font.size': 8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,}
#          'figure.figsize': fig_size}

mpl.rcParams.update(params)


"""
http://nerdjusttyped.blogspot.de/2010/07/type-1-fonts-and-matplotlib-figures.html

Often times when you get a paper accepted for publication you're asked to submit
the whole pdf using only Type 1 fonts and embed them in the document.
I used matplotlib for all my figures and by default it used Type 3 fonts.
In order to switch to Type 1 I had to throw in these lines:"""

mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
#
#




#%%
#FIGURE DEFINITIONS
################################################################################

#Image and object points

fig1 = plt.figure('Image and Object Points',figsize=(fig_width,0.5*fig_width))
ax_image = fig1.add_subplot(121)
ax_image.set_aspect('equal', 'datalim')
ax_object = fig1.add_subplot(122)
ax_object.set_aspect('equal', 'datalim')


#Homography error and condition numbers
fig2 = plt.figure('Effect of point configuration in homography estimation',figsize=(fig_width,fig_width))
ax_cond = fig2.add_subplot(211)
ax_homo_error = fig2.add_subplot(212, sharex = ax_cond)

#Individual Pose errors
fig3 = plt.figure('Effect of point configuration in Pose estimation (Detailed)',figsize=(fig_width,fig_width))
ax_t_error_ippe = fig3.add_subplot(321)
ax_r_error_ippe = fig3.add_subplot(322)
ax_t_error_epnp = fig3.add_subplot(323, sharex = ax_t_error_ippe)
ax_r_error_epnp = fig3.add_subplot(324, sharex = ax_r_error_ippe)
ax_t_error_pnp = fig3.add_subplot(325, sharex = ax_t_error_ippe)
ax_r_error_pnp = fig3.add_subplot(3,2,6, sharex = ax_r_error_ippe)

#Condensated Pose errors
fig4 = plt.figure('Effect of point configuration in Pose estimation (Grouped)',figsize=(fig_width,fig_width))
ax_t_error_all = fig4.add_subplot(211)
ax_r_error_all = fig4.add_subplot(212)


#%%
# FIGURE 1 PLOTS (IMAGE AND OBJECT POINTS)


objectPoints_asarray = np.array([]).reshape(Din['ObjectPoints'][0].shape[0],0)
imagePoints_asarray = np.array([]).reshape(Din['ImagePoints'][0].shape[0],0)

for i in range(len(Din['ObjectPoints'])):
  objectPoints_asarray = np.hstack([objectPoints_asarray,Din['ObjectPoints'][i]])
  imagePoints_asarray = np.hstack([imagePoints_asarray,Din['ImagePoints'][i]])


#PLOT IMAGE POINTS
plt.sca(ax_image)
ax_image.cla()
pl = CircularPlane()
pl.color = 'grey'
cam.plot_plane(pl)
ax_image.plot(imagePoints_asarray[0,0::4][::2],imagePoints_asarray[1,0::4][::2],'-',color = 'blue',)
ax_image.plot(imagePoints_asarray[0,1::4][::2],imagePoints_asarray[1,1::4][::2],'-',color = 'orange',)
ax_image.plot(imagePoints_asarray[0,2::4][::2],imagePoints_asarray[1,2::4][::2],'-',color = 'black',)
ax_image.plot(imagePoints_asarray[0,3::4][::2],imagePoints_asarray[1,3::4][::2],'-',color = 'magenta',)
ax_image.plot(imagePoints_asarray[0,-Din['n']:],imagePoints_asarray[1,-Din['n']:],'o',color = 'red',markersize=3)
ax_image.plot(imagePoints_asarray[0,0:4],imagePoints_asarray[1,0:4],'x',color = 'black',markersize=3)

ax_image.set_xlim(150,500)#cam.img_width)
ax_image.set_ylim(50,450)#cam.img_height)
ax_image.invert_yaxis()
ax_image.locator_params(nbins=5, axis='x')
ax_image.locator_params(nbins=5, axis='y')

#ax_object.set_aspect('equal', 'datalim')
ax_image.set_title('Image Points')


#PLOT OBJECT POINTS
plt.sca(ax_object)
ax_object.cla()
ax_object.plot(objectPoints_asarray[0,0::4][::2],objectPoints_asarray[1,0::4][::2],'-',color = 'blue',)
ax_object.plot(objectPoints_asarray[0,1::4][::2],objectPoints_asarray[1,1::4][::2],'-',color = 'orange',)
ax_object.plot(objectPoints_asarray[0,2::4][::2],objectPoints_asarray[1,2::4][::2],'-',color = 'black',)
ax_object.plot(objectPoints_asarray[0,3::4][::2],objectPoints_asarray[1,3::4][::2],'-',color = 'magenta',)
ax_object.plot(objectPoints_asarray[0,-Din['n']:],objectPoints_asarray[1,-Din['n']:],'o',color = 'red',markersize=3)
ax_object.plot(objectPoints_asarray[0,0:4],objectPoints_asarray[1,0:4],'x',color = 'black',markersize=3)
pl.plot_plane()
ax_object.set_title('Object Points')
ax_object.set_xlim(-pl.radius-0.05,pl.radius+0.05)
ax_object.set_ylim(-pl.radius-0.05,pl.radius+0.05)
ax_object.locator_params(nbins=5, axis='x')
ax_object.locator_params(nbins=5, axis='y')
#ax_object.set_aspect('equal', 'datalim')

fig1.subplots_adjust(left=0.09, right=0.94, wspace=0.35)
fig1.savefig('dynamic-markers-optimal/img/image_control_points.pdf')
#fig1.savefig('dynamic-markers-optimal/img/image_control_points.png', dpi=900)
#fig1.tight_layout()
plt.show()
plt.pause(0.001)

#%%
#################################
#PLOT CONDITION NUMBER AND HOMOGRAPHY ERRORS

# CALCULATE MEAN AND STANDARD DEVIATIONS
t = np.arange(len(Din['homo_dlt_error']))

homo_dlt_error_mean = []
homo_dlt_error_std = []
homo_HO_error_mean = []
homo_HO_error_std = []
homo_CV_error_mean = []
homo_CV_error_std = []

for i in t:
  homo_dlt_error_mean.append(np.mean(Din['homo_dlt_error'][i]))
  homo_dlt_error_std.append(np.std(Din['homo_dlt_error'][i]))

  homo_HO_error_mean.append(np.mean(Din['homo_HO_error'][i]))
  homo_HO_error_std.append(np.std(Din['homo_HO_error'][i]))

  homo_CV_error_mean.append(np.mean(Din['homo_CV_error'][i]))
  homo_CV_error_std.append(np.std(Din['homo_CV_error'][i]))


plt.sca(ax_cond)
ax_cond.cla()
ax_cond.plot(Din['cond_number'],'-', label=r'$c(\mathbf{A}(t))$')

#PLOT HOMO DLT ERROR
plt.sca(ax_homo_error)
ax_homo_error.cla()
mu = np.array(homo_dlt_error_mean)
sigma = np.array(homo_dlt_error_std)
ax_homo_error.plot(t,mu,'-g', label=r'$\mu\left(HE\left(\hat{\mathbf{H}}(t)\right)\right)$')
ax_homo_error.fill_between(t,mu+sigma, mu-sigma, facecolor='green', alpha=0.25, label=r'$\sigma\left(HE\left(\hat{\mathbf{H}}(t)\right)\right)$')
ax_homo_error.set_ylim(0,50)


#PLOT HOMO HO ERROR
#ax_homo_error.plot(homo_HO_error_mean)
#PLOT CV HO ERROR
#mu = homo_CV_error_mean
#sigma = homo_CV_error_std
#ax_homo_error.plot(t,mu)
#ax_homo_error.fill_between(t,mu+sigma, mu-sigma, facecolor='red', alpha=0.5)
#      ax_cond_norm.set_title('Condition number of the Normalized A matrix')

ax_cond.set_title('Evolution of Condition number')
ax_homo_error.set_title('Homography reprojection error (for validation points)')

#ax_cond.set_xlabel('Iterations')
ax_homo_error.set_xlabel('Iterations')
ax_cond.legend(loc='upper rigth')
ax_homo_error.legend(loc='upper rigth')

#ax_homo_error.locator_params(nbins=5, axis='x')
#ax_cond.locator_params(nbins=5, axis='y')


#ax_object.set_aspect('equal', 'datalim')
fig2.tight_layout()
fig2.savefig('dynamic-markers-optimal/img/homography_fronto_parallel.pdf')
#fig2.savefig('dynamic-markers-optimal/img/homography_fronto_parallel.png', dpi=900)

plt.show()
plt.pause(0.001)

#%%
##################################
#PLOT POSE ERRORS

# CALCULATE MEAN AND STANDARD DEVIATIONS
t = np.arange(len(Din['ippe_tvec_error']))

ippe_tvec_error_mean = []
ippe_tvec_error_std = []
ippe_rmat_error_mean = []
ippe_rmat_error_std = []

epnp_tvec_error_mean = []
epnp_tvec_error_std = []
epnp_rmat_error_mean = []
epnp_rmat_error_std = []

pnp_tvec_error_mean = []
pnp_tvec_error_std = []
pnp_rmat_error_mean = []
pnp_rmat_error_std = []

for i in t:
  ippe_tvec_error_mean.append(np.std(Din['ippe_tvec_error'][i]))
  ippe_tvec_error_std.append(np.std(Din['ippe_tvec_error'][i]))
  ippe_rmat_error_mean.append(np.std(Din['ippe_rmat_error'][i]))
  ippe_rmat_error_std.append(np.std(Din['ippe_rmat_error'][i]))

  epnp_tvec_error_mean.append(np.std(Din['epnp_tvec_error'][i]))
  epnp_tvec_error_std.append(np.std(Din['epnp_tvec_error'][i]))
  epnp_rmat_error_mean.append(np.std(Din['epnp_rmat_error'][i]))
  epnp_rmat_error_std.append(np.std(Din['epnp_rmat_error'][i]))

  pnp_tvec_error_mean.append(np.std(Din['pnp_tvec_error'][i]))
  pnp_tvec_error_std.append(np.std(Din['pnp_tvec_error'][i]))
  pnp_rmat_error_mean.append(np.std(Din['pnp_rmat_error'][i]))
  pnp_rmat_error_std.append(np.std(Din['pnp_rmat_error'][i]))




#IPPE
plt.sca(ax_t_error_ippe)
mu = np.array(ippe_tvec_error_mean)
sigma = np.array(ippe_tvec_error_std)
lippe, = ax_t_error_ippe.plot(t,mu,'b', label = '$\mu$')
ax_t_error_ippe.fill_between(t,mu+sigma, mu-sigma, facecolor='blue', alpha=0.25, label = '$\sigma$')
ax_t_error_ippe.set_ylim(0,20)

ax_t_error_all.plot(t,mu,'-b', label = 'mu', lw=1)
#ax_t_error_all.fill_between(t,mu+sigma, mu-sigma, facecolor='blue', alpha=0.25, label = 'sigma')

ax_t_error_ippe.legend(loc='upper rigth')#, prop={'size': 10})
#ax_t_error_ippe.set_ylim(0,5)#semilogy

plt.sca(ax_r_error_ippe)
mu = np.array(ippe_rmat_error_mean)
sigma = np.array(ippe_rmat_error_std)
ax_r_error_ippe.plot(t,mu)
ax_r_error_ippe.fill_between(t,mu+sigma, mu-sigma, facecolor='blue', alpha=0.25)
ax_r_error_ippe.set_ylim(0,35)

ax_r_error_all.plot(t,mu,'-b', lw=1)
#ax_r_error_all.fill_between(t,mu+sigma, mu-sigma, facecolor='blue', alpha=0.1)

#EPNP
plt.sca(ax_t_error_epnp)
mu = np.array(epnp_tvec_error_mean)
sigma = np.array(epnp_tvec_error_std)
lepnp, = ax_t_error_epnp.plot(t,mu,'g')
ax_t_error_epnp.fill_between(t,mu+sigma, mu-sigma, facecolor='green', alpha=0.25)
ax_t_error_epnp.set_ylim(0,20)

ax_t_error_all.plot(t,mu,'-g', lw=1)
#ax_t_error_all.fill_between(t,mu+sigma, mu-sigma, facecolor='green', alpha=0.1)

plt.sca(ax_r_error_epnp)
mu = np.array(epnp_rmat_error_mean)
sigma = np.array(epnp_rmat_error_std)
ax_r_error_epnp.plot(t,mu,'g')
ax_r_error_epnp.fill_between(t,mu+sigma, mu-sigma, facecolor='green', alpha=0.25)
ax_r_error_epnp.set_ylim(0,35)

ax_r_error_all.plot(t,mu,'-g', lw=1)
#ax_r_error_all.fill_between(t,mu+sigma, mu-sigma, facecolor='green', alpha=0.1)

#ITERATIVE PNP
plt.sca(ax_t_error_pnp)
mu = np.array(pnp_tvec_error_mean)
sigma = np.array(pnp_tvec_error_std)
lpnp, = ax_t_error_pnp.plot(t,mu,'r')
ax_t_error_pnp.fill_between(t,mu+sigma, mu-sigma, facecolor='red', alpha=0.25)
ax_t_error_pnp.set_ylim(0,20)

ax_t_error_all.plot(t,mu,'-r', lw=1)
#ax_t_error_all.fill_between(t,mu+sigma, mu-sigma, facecolor='red', alpha=0.1)


plt.sca(ax_r_error_pnp)
mu = np.array(pnp_rmat_error_mean)
sigma = np.array(pnp_rmat_error_std)
ax_r_error_pnp.plot(t,mu,'r')
ax_r_error_pnp.fill_between(t,mu+sigma, mu-sigma, facecolor='red', alpha=0.25)
ax_r_error_pnp.set_ylim(0,35)

ax_r_error_all.plot(t,mu,'-r', lw=1)
#ax_r_error_all.fill_between(t,mu+sigma, mu-sigma, facecolor='red', alpha=0.1)

#Figure Titles



ax_t_error_ippe.set_title(r'$\mathbf{T}$ \textbf{error} (\%)')
ax_r_error_ippe.set_title(r'$\mathbf{R}$ \textbf{error} ($^{\circ}$)')

ax_t_error_all.set_title(r'$\mathbf{T}$ \textbf{error} (\%)')
ax_r_error_all.set_title(r'$\mathbf{R}$ \textbf{error} ($^{\circ}$)')
ax_t_error_all.set_ylabel(r'Percent')
ax_r_error_all.set_ylabel(r'Angle (degrees)')
#ax_t_error_all.set_xlabel('Iterations')
ax_r_error_all.set_xlabel('Iterations')


ax_t_error_pnp.set_xlabel('Iterations')
ax_r_error_pnp.set_xlabel('Iterations')


#ax_t_error_epnp.set_title('Translation error (in percent) for EPnP Pose')
#ax_r_error_epnp.set_title('Rotation error (Angle) for EPnP Pose')
#ax_t_error_pnp.set_title('Translation error (in percent) for PnP Pose')
#ax_r_error_pnp.set_title('Rotation error (Angle) for PnP Pose')

#fig3.legend((lippe, sippe), ('Line 1', 'Line 2'), 'lower center',ncol=2, mode="expand", borderaxespad=0.)

fig3.legend([lippe, lepnp, lpnp], ['IPPE', 'EPnP', 'LM'], 'lower center',ncol=3)
ax_t_error_all.legend([lippe, lepnp, lpnp], ['IPPE', 'EPnP', 'LM'])
#fig3.legend([lippe], ['IPPE'], 'lower center')

#, 'lower center',ncol=2, mode="expand", borderaxespad=0.)

#plt.figlegend( lines, labels, loc = 'lower center', ncol=1, labelspacing=0. )

plt.setp(ax_homo_error.get_xticklabels(), visible=False)
plt.setp(ax_cond.get_xticklabels(), visible=False)
plt.setp(ax_t_error_ippe.get_xticklabels(), visible=False)
plt.setp(ax_r_error_ippe.get_xticklabels(), visible=False)
plt.setp(ax_t_error_epnp.get_xticklabels(), visible=False)
plt.setp(ax_r_error_epnp.get_xticklabels(), visible=False)


fig3.tight_layout()
fig3.subplots_adjust(bottom=0.20)
fig3.savefig('dynamic-markers-optimal/img/pose_separate_fronto_parallel.pdf')
#fig3.savefig('dynamic-markers-optimal/img/pose_separate_fronto_parallel.png', dpi=900)

fig4.tight_layout()
fig4.savefig('dynamic-markers-optimal/img/pose_together_fronto_parallel.pdf')
#fig4.savefig('dynamic-markers-optimal/img/pose_together_fronto_parallel.png', dpi=900)



#plt.tight_layout()
plt.show()
plt.pause(0.001)
