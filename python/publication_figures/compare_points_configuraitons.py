# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 22:45:23 2017

@author: racuna
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

mpl.pyplot.close("all")
#mpl.rc('font',**{'family':'serif','serif':['Times']})
#mpl.rc('text', usetex=True)


Din = pickle.load( open( "icra_sim1_inclined.p", "rb" ) )
cam = Din["Camera"]
pl = Din['Plane']



fig_width_pt = 245.71811  # Get this from LaTeX using \showthe\columnwidth

inch_per_cent = 0.393701
inches_per_pt = 1.0/72.27               # Convert pt to inch
#golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = 10*inch_per_cent      # height in inches
#fig_size =  [fig_width,fig_height]
#params = {'backend': 'PDF',
#          'axes.labelsize': 8,
#          'text.fontsize': 8,
#          'legend.fontsize': 8,
#          'xtick.labelsize': 8,
#          'ytick.labelsize': 8,
#          'text.usetex': True,}
##          'figure.figsize': fig_size}
#mpl.rcParams.update(params)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
#
#

class DataSimOut(object):
  def __init__(self,Npoints,Plane,ValidationPlane, ImageNoise = 4, ValidationIters = 100):
    self.n = Npoints
    self.Plane = Plane
    self.ValidationPlane = ValidationPlane
    self.Camera = []  
    self.ObjectPoints = []
    self.ImagePoints = []
    self.CondNumber = []
    self.CondNumberNorm = []
    
    self.Homo_DLT_mean = []
    self.Homo_HO_mean = []
    self.Homo_CV_mean = []
    self.ippe_tvec_error_mean = []
    self.ippe_rmat_error_mean = []
    self.epnp_tvec_error_mean = []
    self.epnp_rmat_error_mean = []
    self.pnp_tvec_error_mean = []
    self.pnp_rmat_error_mean = []
    
    self.Homo_DLT_std = []
    self.Homo_HO_std = []
    self.Homo_CV_std = []
    self.ippe_tvec_error_std = []
    self.ippe_rmat_error_std = []
    self.epnp_tvec_error_std = []
    self.epnp_rmat_error_std = []
    self.pnp_tvec_error_std = []
    self.pnp_rmat_error_std = []
    
    
    self.ValidationIters = 100
    self.ImageNoise = ImageNoise
    
  def calculate_metrics(self):
    new_objectPoints = self.ObjectPoints[-1]
    cam = self.Camera[-1]
    validation_plane = self.ValidationPlane
    new_imagePoints = np.array(cam.project(new_objectPoints, False))
    self.ImagePoints.append(new_imagePoints)
    #CONDITION NUMBER CALCULATION
    input_list = gd.extract_objectpoints_vars(new_objectPoints)
    input_list.append(np.array(cam.P))    
    mat_cond = gd.matrix_condition_number_autograd(*input_list, normalize = False)
   
    #CONDITION NUMBER WITH A NORMALIZED CALCULATION
    input_list = gd.extract_objectpoints_vars(new_objectPoints)
    input_list.append(np.array(cam.P))
    mat_cond_normalized = gd.matrix_condition_number_autograd(*input_list, normalize = True)
  
    self.CondNumber.append(mat_cond)
    self.CondNumberNorm.append(mat_cond_normalized)
  
    ##HOMOGRAPHY ERRORS
    ## TRUE VALUE OF HOMOGRAPHY OBTAINED FROM CAMERA PARAMETERS
    Hcam = cam.homography_from_Rt()
    ##We add noise to the image points and calculate the noisy homography
    homo_dlt_error_loop = []
    homo_HO_error_loop = []
    homo_CV_error_loop = []
    ippe_tvec_error_loop = []
    ippe_rmat_error_loop = []
    epnp_tvec_error_loop = []
    epnp_rmat_error_loop = []
    pnp_tvec_error_loop = []
    pnp_rmat_error_loop = []
    
    
    # WE CREATE NOISY IMAGE POINTS (BASED ON THE TRUE VALUES) AND CALCULATE
    # THE ERRORS WE THEN OBTAIN AN AVERAGE FOR EACH ONE
    for j in range(self.ValidationIters):
      new_imagePoints_noisy = cam.addnoise_imagePoints(new_imagePoints, mean = 0, sd = self.ImageNoise)
    
      #Calculate the pose using IPPE (solution with least repro error)
      normalizedimagePoints = cam.get_normalized_pixel_coordinates(new_imagePoints_noisy)
      ippe_tvec1, ippe_rmat1, ippe_tvec2, ippe_rmat2 = pose_ippe_both(new_objectPoints, normalizedimagePoints, debug = False)
      ippeCam1 = cam.clone_withPose(ippe_tvec1, ippe_rmat1)
  
      #Calculate the pose using solvepnp EPNP
      debug = False
      epnp_tvec, epnp_rmat = pose_pnp(new_objectPoints, new_imagePoints_noisy, cam.K, debug, cv2.SOLVEPNP_EPNP,False)
      epnpCam = cam.clone_withPose(epnp_tvec, epnp_rmat)
  
      #Calculate the pose using solvepnp ITERATIVE
      pnp_tvec, pnp_rmat = pose_pnp(new_objectPoints, new_imagePoints_noisy, cam.K, debug, cv2.SOLVEPNP_ITERATIVE,False)
      pnpCam = cam.clone_withPose(pnp_tvec, pnp_rmat)
  
      #Calculate errors
      ippe_tvec_error1, ippe_rmat_error1 = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, ippeCam1.get_tvec(), ippe_rmat1)
      ippe_tvec_error_loop.append(ippe_tvec_error1)
      ippe_rmat_error_loop.append(ippe_rmat_error1)        
  
      epnp_tvec_error, epnp_rmat_error = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, epnpCam.get_tvec(), epnp_rmat)
      epnp_tvec_error_loop.append(epnp_tvec_error)
      epnp_rmat_error_loop.append(epnp_rmat_error)
  
      pnp_tvec_error, pnp_rmat_error = ef.calc_estimated_pose_error(cam.get_tvec(), cam.R, pnpCam.get_tvec(), pnp_rmat)
      pnp_tvec_error_loop.append(pnp_tvec_error)
      pnp_rmat_error_loop.append(pnp_rmat_error) 
  
      #Homography Estimation from noisy image points
  
      #DLT TRANSFORM
      Xo = new_objectPoints[[0,1,3],:]
      Xi = new_imagePoints_noisy
      Hnoisy_dlt,_,_ = homo2d.homography2d(Xo,Xi)
      Hnoisy_dlt = Hnoisy_dlt/Hnoisy_dlt[2,2]
      
      #HO METHOD
      Xo = new_objectPoints[[0,1,3],:]
      Xi = new_imagePoints_noisy
      Hnoisy_HO = hh(Xo,Xi)
      
      #OpenCV METHOD
      Xo = new_objectPoints[[0,1,3],:]
      Xi = new_imagePoints_noisy
      Hnoisy_OpenCV,_ = cv2.findHomography(Xo[:2].T.reshape(1,-1,2),Xi[:2].T.reshape(1,-1,2))
      
  
      ## ERRORS FOR THE  DLT HOMOGRAPHY 
      ## VALIDATION OBJECT POINTS
      validation_objectPoints =validation_plane.get_points()
      validation_imagePoints = np.array(cam.project(validation_objectPoints, False))
      Xo = np.copy(validation_objectPoints)
      Xo = np.delete(Xo, 2, axis=0)
      Xi = np.copy(validation_imagePoints)
      homo_dlt_error_loop.append(ef.validation_points_error(Xi, Xo, Hnoisy_dlt))
      
      ## ERRORS FOR THE  HO HOMOGRAPHY 
      ## VALIDATION OBJECT POINTS
      validation_objectPoints =validation_plane.get_points()
      validation_imagePoints = np.array(cam.project(validation_objectPoints, False))
      Xo = np.copy(validation_objectPoints)
      Xo = np.delete(Xo, 2, axis=0)
      Xi = np.copy(validation_imagePoints)        
      homo_HO_error_loop.append(ef.validation_points_error(Xi, Xo, Hnoisy_HO))
      
      ## ERRORS FOR THE  OpenCV HOMOGRAPHY 
      ## VALIDATION OBJECT POINTS
      validation_objectPoints =validation_plane.get_points()
      validation_imagePoints = np.array(cam.project(validation_objectPoints, False))
      Xo = np.copy(validation_objectPoints)
      Xo = np.delete(Xo, 2, axis=0)
      Xi = np.copy(validation_imagePoints)        
      homo_CV_error_loop.append(ef.validation_points_error(Xi, Xo, Hnoisy_OpenCV))
  
    
    self.Homo_DLT_mean.append(np.mean(homo_dlt_error_loop))
    self.Homo_HO_mean.append(np.mean(homo_HO_error_loop))
    self.Homo_CV_mean.append(np.mean(homo_CV_error_loop))
    self.ippe_tvec_error_mean.append(np.mean(ippe_tvec_error_loop))
    self.ippe_rmat_error_mean.append(np.mean(ippe_rmat_error_loop))   
    self.epnp_tvec_error_mean.append(np.mean(epnp_tvec_error_loop))
    self.epnp_rmat_error_mean.append(np.mean(epnp_rmat_error_loop))    
    self.pnp_tvec_error_mean.append(np.mean(pnp_tvec_error_loop))
    self.pnp_rmat_error_mean.append(np.mean(pnp_rmat_error_loop))
    
    self.Homo_DLT_std.append(np.std(homo_dlt_error_loop))
    self.Homo_HO_std.append(np.std(homo_HO_error_loop))
    self.Homo_CV_std.append(np.std(homo_CV_error_loop))
    self.ippe_tvec_error_std.append(np.std(ippe_tvec_error_loop))
    self.ippe_rmat_error_std.append(np.std(ippe_rmat_error_loop))   
    self.epnp_tvec_error_std.append(np.std(epnp_tvec_error_loop))
    self.epnp_rmat_error_std.append(np.std(epnp_rmat_error_loop))
    self.pnp_tvec_error_std.append(np.std(pnp_tvec_error_loop))
    self.pnp_rmat_error_std.append(np.std(pnp_rmat_error_loop))

def remove_invalid(data, limit = 100):
  data_out = np.array(data)
  data_out = data_out[data_out < limit]
  data_out = data_out[np.isfinite(data_out)]
  return data_out

def remove_invalid_cond(data):
  data_out = np.array(data)
  data_out = data_out[data_out < 100]
  data_out = data_out[np.isfinite(data_out)]
  return data_out

#OPEN SAVED DATA
[D4pIll,D4pWell] = pickle.load( open( "08.09.2017/icra_sim_illvsWell_4points.p", "rb" ) )
[D5pIll,D5pWell] = pickle.load( open( "08.09.2017/icra_sim_illvsWell_5points.p", "rb" ) )
[D6pIll,D6pWell] = pickle.load( open( "08.09.2017/icra_sim_illvsWell_6points.p", "rb" ) )
[D7pIll,D7pWell] = pickle.load( open( "08.09.2017/icra_sim_illvsWell_7points.p", "rb" ) )
[D8pIll,D8pWell] = pickle.load( open( "08.09.2017/icra_sim_illvsWell_8points.p", "rb" ) )
[D9pIll,D10pIll,D11pIll,D12pIll,D13pIll,D14pIll,D15pIll,D16pIll] = pickle.load( open( "12.09.2017/icra_sim_ill_only_9-16points.p", "rb" ) )


#CREATE FIGURES
#figure for Homography error and condition numbers
fig2 = plt.figure('Well conditioned Vs Ill conditioned configurations (Homography)',figsize=(fig_width,0.7*fig_width))
ax_cond = fig2.add_subplot(211)    
ax_homo_error = fig2.add_subplot(212, sharex = ax_cond)
ax_homo_error.set_xticks([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

#another figure for individual Pose errors
fig3 = plt.figure('Well conditioned Vs Ill conditioned configurations (Pose)',figsize=(fig_width,1.1*fig_width))
ax_t_error_ippe = fig3.add_subplot(321)
ax_r_error_ippe = fig3.add_subplot(322)
ax_t_error_epnp = fig3.add_subplot(323, sharex = ax_t_error_ippe)
ax_r_error_epnp = fig3.add_subplot(324, sharex = ax_r_error_ippe)    
ax_t_error_pnp = fig3.add_subplot(325, sharex = ax_t_error_ippe)
ax_r_error_pnp = fig3.add_subplot(3,2,6, sharex = ax_r_error_ippe)

ax_t_error_pnp.set_xticks([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
ax_r_error_pnp.set_xticks([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

#CONDITION NUMBER
mean4Well = np.median((D4pWell.CondNumber))
mean5Well = np.median((D5pWell.CondNumber))
mean6Well = np.median((D6pWell.CondNumber))
mean7Well = np.median((D7pWell.CondNumber))
mean8Well = np.median((D8pWell.CondNumber))

mean4Ill = np.median((D4pIll.CondNumber))
mean5Ill = np.median((D5pIll.CondNumber))
mean6Ill = np.median((D6pIll.CondNumber))
mean7Ill = np.median((D7pIll.CondNumber))
mean8Ill = np.median((D8pIll.CondNumber))
mean9Ill = np.median((D9pIll.CondNumber))
mean10Ill = np.median((D10pIll.CondNumber))
mean11Ill = np.median((D11pIll.CondNumber))
mean12Ill = np.median((D12pIll.CondNumber))
mean13Ill = np.median((D13pIll.CondNumber))
mean14Ill = np.median((D14pIll.CondNumber))
mean15Ill = np.median((D15pIll.CondNumber))
mean16Ill = np.median((D16pIll.CondNumber))

t = np.arange(4,9)
ax_cond.semilogy(t,[mean4Well, mean5Well, mean6Well, mean7Well, mean8Well],'-b|',label = "Well-Conditioned")
t2 = np.arange(4,15)
ax_cond.semilogy(t2,[mean4Ill, mean5Ill, mean6Ill, mean7Ill, mean8Ill,
                    mean9Ill,mean10Ill,mean11Ill,mean12Ill,mean13Ill,mean14Ill],'-r|', label = "Ill-Conditioned")
ax_cond.axhline(y=mean8Well, color='black', linestyle=':')
#percentage_improvement = (new - old)/old*100%

#per4 = (mean4Well)/mean4Ill*100
#per5 = (mean5Well)/mean5Ill*100
#per6 = (mean6Well)/mean6Ill*100
#per7 = (mean7Well)/mean7Ill*100
#per8 = (mean8Well)/mean8Ill*100

#ax_cond.plot(t,[per4, per5, per6, per7, per8],'-x')

#HOMOGRAPHY
mean4Well = np.mean(remove_invalid(D4pWell.Homo_DLT_mean))
mean5Well = np.mean(remove_invalid(D5pWell.Homo_DLT_mean))
mean6Well = np.mean(remove_invalid(D6pWell.Homo_DLT_mean))
mean7Well = np.mean(remove_invalid(D7pWell.Homo_DLT_mean))
mean8Well = np.mean(remove_invalid(D8pWell.Homo_DLT_mean))

mean4Ill = np.mean(remove_invalid(D4pIll.Homo_DLT_mean))
mean5Ill = np.mean(remove_invalid(D5pIll.Homo_DLT_mean))
mean6Ill = np.mean(remove_invalid(D6pIll.Homo_DLT_mean))
mean7Ill = np.mean(remove_invalid(D7pIll.Homo_DLT_mean))
mean8Ill = np.mean(remove_invalid(D8pIll.Homo_DLT_mean))
mean9Ill = np.mean(remove_invalid(D9pIll.Homo_DLT_mean))
mean10Ill = np.mean(remove_invalid(D10pIll.Homo_DLT_mean))
mean11Ill = np.mean(remove_invalid(D11pIll.Homo_DLT_mean))
mean12Ill = np.mean(remove_invalid(D12pIll.Homo_DLT_mean))
mean13Ill = np.mean(remove_invalid(D13pIll.Homo_DLT_mean))
mean14Ill = np.mean(remove_invalid(D14pIll.Homo_DLT_mean))
mean15Ill = np.mean(remove_invalid(D15pIll.Homo_DLT_mean))
mean16Ill = np.mean(remove_invalid(D16pIll.Homo_DLT_mean))


#ax_homo_error.plot(t,[mean4Ill, mean5Ill, mean6Ill, mean7Ill, mean8Ill],'-rx', label = "Ill-Conditioned")
ax_homo_error.plot(t2,[mean4Ill, mean5Ill, mean6Ill, mean7Ill, mean8Ill,
                    mean9Ill,mean10Ill,mean11Ill,mean12Ill,mean13Ill,mean14Ill],'-r|', label = "Ill-Conditioned")
ax_homo_error.plot(t,[mean4Well, mean5Well, mean6Well, mean7Well, mean8Well],'-b|',label = "Well-Conditioned")
ax_homo_error.axhline(y=mean4Well, color='black', linestyle='-.')
ax_homo_error.axhline(y=mean8Well, color='black', linestyle=':')



ax_cond.set_title('Condition number')
ax_homo_error.set_title('Homography error')


ax_homo_error.set_xlabel('number of points')
#ax_cond.legend(loc='upper rigth')
#ax_homo_error.legend(loc='upper rigth')

plt.setp(ax_cond.get_xticklabels(), visible=False)
plt.tight_layout()

plt.show()
####################################################################################
#IPPE
#Translation
mean4Well = np.mean(remove_invalid(D4pWell.ippe_tvec_error_mean))
mean5Well = np.mean(remove_invalid(D5pWell.ippe_tvec_error_mean))
mean6Well = np.mean(remove_invalid(D6pWell.ippe_tvec_error_mean))
mean7Well = np.mean(remove_invalid(D7pWell.ippe_tvec_error_mean))
mean8Well = np.mean(remove_invalid(D8pWell.ippe_tvec_error_mean))

mean4Ill = np.mean(remove_invalid(D4pIll.ippe_tvec_error_mean))
mean5Ill = np.mean(remove_invalid(D5pIll.ippe_tvec_error_mean))
mean6Ill = np.mean(remove_invalid(D6pIll.ippe_tvec_error_mean))
mean7Ill = np.mean(remove_invalid(D7pIll.ippe_tvec_error_mean))
mean8Ill = np.mean(remove_invalid(D8pIll.ippe_tvec_error_mean))
mean9Ill = np.mean(remove_invalid(D9pIll.ippe_tvec_error_mean))
mean10Ill = np.mean(remove_invalid(D10pIll.ippe_tvec_error_mean))
mean11Ill = np.mean(remove_invalid(D11pIll.ippe_tvec_error_mean))
mean12Ill = np.mean(remove_invalid(D12pIll.ippe_tvec_error_mean))
mean13Ill = np.mean(remove_invalid(D13pIll.ippe_tvec_error_mean))
#mean14Ill = np.mean(remove_invalid(D14pIll.ippe_tvec_error_mean))
#mean15Ill = np.mean(remove_invalid(D15pIll.ippe_tvec_error_mean))
#mean16Ill = np.mean(remove_invalid(D16pIll.ippe_tvec_error_mean))


#Ill_cond, = ax_t_error_ippe.plot(t,[mean4Ill, mean5Ill, mean6Ill, mean7Ill, mean8Ill],'-rx', label = "Ill-Conditioned")
Ill_cond, = ax_t_error_ippe.plot(t2,[mean4Ill, mean5Ill, mean6Ill, mean7Ill, mean8Ill,
                    mean9Ill,mean10Ill,mean11Ill,mean12Ill,mean13Ill,mean14Ill],'-r|', label = "Ill-Conditioned")
Well_cond, = ax_t_error_ippe.plot(t,[mean4Well, mean5Well, mean6Well, mean7Well, mean8Well],'-b|',label = "Well-Conditioned")

ax_t_error_ippe.axhline(y=mean4Well, color='black', linestyle='-.')

#Rotation
mean4Well = np.mean(remove_invalid(D4pWell.ippe_rmat_error_mean))
mean5Well = np.mean(remove_invalid(D5pWell.ippe_rmat_error_mean))
mean6Well = np.mean(remove_invalid(D6pWell.ippe_rmat_error_mean))
mean7Well = np.mean(remove_invalid(D7pWell.ippe_rmat_error_mean))
mean8Well = np.mean(remove_invalid(D8pWell.ippe_rmat_error_mean))

mean4Ill = np.mean(remove_invalid(D4pIll.ippe_rmat_error_mean))
mean5Ill = np.mean(remove_invalid(D5pIll.ippe_rmat_error_mean))
mean6Ill = np.mean(remove_invalid(D6pIll.ippe_rmat_error_mean))
mean7Ill = np.mean(remove_invalid(D7pIll.ippe_rmat_error_mean))
mean8Ill = np.mean(remove_invalid(D8pIll.ippe_rmat_error_mean))
mean9Ill = np.mean(remove_invalid(D9pIll.ippe_rmat_error_mean))
mean10Ill = np.mean(remove_invalid(D10pIll.ippe_rmat_error_mean))
mean11Ill = np.mean(remove_invalid(D11pIll.ippe_rmat_error_mean))
mean12Ill = np.mean(remove_invalid(D12pIll.ippe_rmat_error_mean))
mean13Ill = np.mean(remove_invalid(D13pIll.ippe_rmat_error_mean))
mean14Ill = np.mean(remove_invalid(D14pIll.ippe_rmat_error_mean))
mean15Ill = np.mean(remove_invalid(D15pIll.ippe_rmat_error_mean))
mean16Ill = np.mean(remove_invalid(D16pIll.ippe_rmat_error_mean))

#ax_r_error_ippe.plot(t,[mean4Ill, mean5Ill, mean6Ill, mean7Ill, mean8Ill],'-rx')
ax_r_error_ippe.plot(t2,[mean4Ill, mean5Ill, mean6Ill, mean7Ill, mean8Ill,
                    mean9Ill,mean10Ill,mean11Ill,mean12Ill,mean13Ill,mean14Ill],'-r|', label = "Ill-Conditioned")
ax_r_error_ippe.plot(t,[mean4Well, mean5Well, mean6Well, mean7Well, mean8Well],'-b|')

ax_r_error_ippe.axhline(y=mean4Well, color='black', linestyle='-.')

#EPnP
#Translation
mean4Well = np.mean(remove_invalid(D4pWell.epnp_tvec_error_mean))
mean5Well = np.mean(remove_invalid(D5pWell.epnp_tvec_error_mean))
mean6Well = np.mean(remove_invalid(D6pWell.epnp_tvec_error_mean))
mean7Well = np.mean(remove_invalid(D7pWell.epnp_tvec_error_mean))
mean8Well = np.mean(remove_invalid(D8pWell.epnp_tvec_error_mean))

mean4Ill = np.mean(remove_invalid(D4pIll.epnp_tvec_error_mean))
mean5Ill = np.mean(remove_invalid(D5pIll.epnp_tvec_error_mean))
mean6Ill = np.mean(remove_invalid(D6pIll.epnp_tvec_error_mean))
mean7Ill = np.mean(remove_invalid(D7pIll.epnp_tvec_error_mean))
mean8Ill = np.mean(remove_invalid(D8pIll.epnp_tvec_error_mean))
mean9Ill = np.mean(remove_invalid(D9pIll.epnp_tvec_error_mean))
mean10Ill = np.mean(remove_invalid(D10pIll.epnp_tvec_error_mean))
mean11Ill = np.mean(remove_invalid(D11pIll.epnp_tvec_error_mean))
mean12Ill = np.mean(remove_invalid(D12pIll.epnp_tvec_error_mean))
mean13Ill = np.mean(remove_invalid(D13pIll.epnp_tvec_error_mean))
mean14Ill = np.mean(remove_invalid(D14pIll.epnp_tvec_error_mean))
mean15Ill = np.mean(remove_invalid(D15pIll.epnp_tvec_error_mean))
mean16Ill = np.mean(remove_invalid(D16pIll.epnp_tvec_error_mean))

ax_t_error_epnp.plot(t2,[mean4Ill, mean5Ill, mean6Ill, mean7Ill, mean8Ill,
                    mean9Ill,mean10Ill,mean11Ill,mean12Ill,mean13Ill,mean14Ill],'-r|', label = "Ill-Conditioned")
ax_t_error_epnp.plot(t,[mean4Well, mean5Well, mean6Well, mean7Well, mean8Well],'-b|')

ax_t_error_epnp.axhline(y=mean4Well, color='black', linestyle='-.')

#Rotation
mean4Well = np.mean(remove_invalid(D4pWell.epnp_rmat_error_mean))
mean5Well = np.mean(remove_invalid(D5pWell.epnp_rmat_error_mean))
mean6Well = np.mean(remove_invalid(D6pWell.epnp_rmat_error_mean))
mean7Well = np.mean(remove_invalid(D7pWell.epnp_rmat_error_mean))
mean8Well = np.mean(remove_invalid(D8pWell.epnp_rmat_error_mean))

mean4Ill = np.mean(remove_invalid(D4pIll.epnp_rmat_error_mean))
mean5Ill = np.mean(remove_invalid(D5pIll.epnp_rmat_error_mean))
mean6Ill = np.mean(remove_invalid(D6pIll.epnp_rmat_error_mean))
mean7Ill = np.mean(remove_invalid(D7pIll.epnp_rmat_error_mean))
mean8Ill = np.mean(remove_invalid(D8pIll.epnp_rmat_error_mean))
mean9Ill = np.mean(remove_invalid(D9pIll.epnp_rmat_error_mean))
mean10Ill = np.mean(remove_invalid(D10pIll.epnp_rmat_error_mean))
mean11Ill = np.mean(remove_invalid(D11pIll.epnp_rmat_error_mean))
mean12Ill = np.mean(remove_invalid(D12pIll.epnp_rmat_error_mean))
mean13Ill = np.mean(remove_invalid(D13pIll.epnp_rmat_error_mean))
mean14Ill = np.mean(remove_invalid(D14pIll.epnp_rmat_error_mean))
mean15Ill = np.mean(remove_invalid(D15pIll.epnp_rmat_error_mean))
mean16Ill = np.mean(remove_invalid(D16pIll.epnp_rmat_error_mean))



ax_r_error_epnp.plot(t2,[mean4Ill, mean5Ill, mean6Ill, mean7Ill, mean8Ill,
                    mean9Ill,mean10Ill,mean11Ill,mean12Ill,mean13Ill,mean14Ill],'-r|', label = "Ill-Conditioned")
ax_r_error_epnp.plot(t,[mean4Well, mean5Well, mean6Well, mean7Well, mean8Well],'-b|')

ax_r_error_epnp.axhline(y=mean4Well, color='black', linestyle='-.')

#PnP
#Translation
mean4Well = np.mean(remove_invalid(D4pWell.pnp_tvec_error_mean))
mean5Well = np.mean(remove_invalid(D5pWell.pnp_tvec_error_mean))
mean6Well = np.mean(remove_invalid(D6pWell.pnp_tvec_error_mean))
mean7Well = np.mean(remove_invalid(D7pWell.pnp_tvec_error_mean))
mean8Well = np.mean(remove_invalid(D8pWell.pnp_tvec_error_mean))

mean4Ill = np.mean(remove_invalid(D4pIll.pnp_tvec_error_mean))
mean5Ill = np.mean(remove_invalid(D5pIll.pnp_tvec_error_mean))
mean6Ill = np.mean(remove_invalid(D6pIll.pnp_tvec_error_mean))
mean7Ill = np.mean(remove_invalid(D7pIll.pnp_tvec_error_mean))
mean8Ill = np.mean(remove_invalid(D8pIll.pnp_tvec_error_mean))
mean9Ill = np.mean(remove_invalid(D9pIll.pnp_tvec_error_mean))
mean10Ill = np.mean(remove_invalid(D10pIll.pnp_tvec_error_mean))
mean11Ill = np.mean(remove_invalid(D11pIll.pnp_tvec_error_mean))
mean12Ill = np.mean(remove_invalid(D12pIll.pnp_tvec_error_mean))
mean13Ill = np.mean(remove_invalid(D13pIll.pnp_tvec_error_mean))
mean14Ill = np.mean(remove_invalid(D14pIll.pnp_tvec_error_mean))
mean15Ill = np.mean(remove_invalid(D15pIll.pnp_tvec_error_mean))
mean16Ill = np.mean(remove_invalid(D16pIll.pnp_tvec_error_mean))

ax_t_error_pnp.plot(t2,[mean4Ill, mean5Ill, mean6Ill, mean7Ill, mean8Ill,
                    mean9Ill,mean10Ill,mean11Ill,mean12Ill,mean13Ill,mean14Ill],'-r|', label = "Ill-Conditioned")
ax_t_error_pnp.plot(t,[mean4Well, mean5Well, mean6Well, mean7Well, mean8Well],'-b|')

ax_t_error_pnp.axhline(y=mean4Well, color='black', linestyle='-.')

#Rotation
mean4Well = np.mean(remove_invalid(D4pWell.pnp_rmat_error_mean))
mean5Well = np.mean(remove_invalid(D5pWell.pnp_rmat_error_mean))
mean6Well = np.mean(remove_invalid(D6pWell.pnp_rmat_error_mean))
mean7Well = np.mean(remove_invalid(D7pWell.pnp_rmat_error_mean))
mean8Well = np.mean(remove_invalid(D8pWell.pnp_rmat_error_mean))

mean4Ill = np.mean(remove_invalid(D4pIll.pnp_rmat_error_mean))
mean5Ill = np.mean(remove_invalid(D5pIll.pnp_rmat_error_mean))
mean6Ill = np.mean(remove_invalid(D6pIll.pnp_rmat_error_mean))
mean7Ill = np.mean(remove_invalid(D7pIll.pnp_rmat_error_mean))
mean8Ill = np.mean(remove_invalid(D8pIll.pnp_rmat_error_mean))
mean9Ill = np.mean(remove_invalid(D9pIll.pnp_rmat_error_mean))
mean10Ill = np.mean(remove_invalid(D10pIll.pnp_rmat_error_mean))
mean11Ill = np.mean(remove_invalid(D11pIll.pnp_rmat_error_mean))
mean12Ill = np.mean(remove_invalid(D12pIll.pnp_rmat_error_mean))
mean13Ill = np.mean(remove_invalid(D13pIll.pnp_rmat_error_mean))
mean14Ill = np.mean(remove_invalid(D14pIll.pnp_rmat_error_mean))
mean15Ill = np.mean(remove_invalid(D15pIll.pnp_rmat_error_mean))
mean16Ill = np.mean(remove_invalid(D16pIll.pnp_rmat_error_mean))


ax_r_error_pnp.plot(t2,[mean4Ill, mean5Ill, mean6Ill, mean7Ill, mean8Ill,
                    mean9Ill,mean10Ill,mean11Ill,mean12Ill,mean13Ill,mean14Ill],'-r|', label = "Ill-Conditioned")
ax_r_error_pnp.plot(t,[mean4Well, mean5Well, mean6Well, mean7Well, mean8Well],'-b|')

ax_r_error_pnp.axhline(y=mean4Well, color='black', linestyle='-.')

ax_t_error_ippe.set_title(r'\textbf{IPPE} $\mathbf{T}$ \textbf{error} (\%)')
ax_r_error_ippe.set_title(r'\textbf{IPPE} $\mathbf{R}$ \textbf{error} ($^{\circ}$)')

ax_t_error_epnp.set_title(r'\textbf{EPnP} $\mathbf{T}$ \textbf{error} (\%)')
ax_r_error_epnp.set_title(r'\textbf{EPnP} $\mathbf{R}$ \textbf{error} ($^{\circ}$)')

ax_t_error_pnp.set_title(r'\textbf{LM} $\mathbf{T}$ \textbf{error} (\%)')
ax_r_error_pnp.set_title(r'\textbf{LM} $\mathbf{R}$ \textbf{error} ($^{\circ}$)')



ax_t_error_pnp.set_xlabel('number of points')
ax_r_error_pnp.set_xlabel('number of points')

fig2.legend([Well_cond, Ill_cond], ['Well-Conditioned', 'Ill-Conditioned'], 'lower center',ncol=2)
fig3.legend([Well_cond, Ill_cond], ['Well-Conditioned', 'Ill-Conditioned'], 'lower center',ncol=2)
#ax_t_error_all.legend([lippe, lepnp, lpnp], ['IPPE', 'EPnP', 'OpenCV SolvePnP'])

plt.setp(ax_t_error_ippe.get_xticklabels(), visible=False)
plt.setp(ax_r_error_ippe.get_xticklabels(), visible=False)      
plt.setp(ax_t_error_epnp.get_xticklabels(), visible=False)
plt.setp(ax_r_error_epnp.get_xticklabels(), visible=False)


fig2.tight_layout()
fig2.subplots_adjust(bottom=0.25)
fig2.savefig('dynamic-markers-optimal/img/point_config_comp_homo.pdf')
fig2.savefig('dynamic-markers-optimal/img/point_config_comp_homo.png', dpi=900)

fig3.tight_layout()
fig3.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.18)
fig3.savefig('dynamic-markers-optimal/img/point_config_comp_pose.pdf')
fig3.savefig('dynamic-markers-optimal/img/point_config_comp_pose.png', dpi=900)
