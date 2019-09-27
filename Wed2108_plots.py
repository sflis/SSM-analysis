#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:42:24 2019

@author: wolkerst
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
import time
import pickle
import os
from scipy.signal import find_peaks
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    

from tqdm import tqdm
from ssm.core import pchain
from ssm.pmodules import *
from ssm.core.util_pmodules import Aggregate
from ssm.pmodules import *

#from ssm.core.pchain import ProcessingModule
from ssm.pmodules import Reader
from CHECLabPy.plotting.camera import CameraImage
from pylab import *
from scipy import optimize
from scipy.optimize import curve_fit

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter,savgol_coeffs
from scipy import interpolate
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
#import ROOT
#from ROOT import TFitResultPtr

folder = []
directory = '/data/CHECS-data/nsbcal/'
for filename in os.listdir(directory):
    if filename.endswith(".hdf5"):
        folder.append(filename)    
folder

number_pixels = 2048
bad_pixel = np.array([25, 58, 96,97,98,99, 101, 226, 247, 256, 259, 304, 448, 449,570, 653, 670, 776, 1049, 1094,
                      1158, 1177,1185, 1212, 1352, 1367, 1381,1427, 1434,1439,1503, 1562, 1680, 1765,
                      1812, 1813, 1814, 1815, 1829, 1852, 1869, 1945, 1957, 2009, 2043])
    
full_inter = []
full_peak_pos = []
full_peak_val = []
full_errors = []
full_number_bins = []
for file in range(10,13):

#    # Choose file in folder
    file = folder[file]
    property_path = os.path.join('/home/wolkerst/projects/cta/SSM-analysis/','40steps_'+file[0:-5])

    with open('40steps_'+file[0:-5], 'rb') as handle:
        get_saved_file = pickle.load(handle)
        
    
    pxl = get_saved_file['Good_pixel']
    inter = get_saved_file['Interval_begin+end']
    inter_ext = get_saved_file["Extended_interval_begin+end"]
    peak_pos = get_saved_file['Position_of_Peaks_in_time']
    peak_val = get_saved_file['Value_of_Peaks']
    number_bins = int(len(inter)/2)
    error = get_saved_file['Error']


    full_number_bins.append(number_bins)
    full_errors.append(error)
    full_inter.append(inter)
    full_peak_pos.append(peak_pos)
    full_peak_val.append(peak_val)
################################################################################    

# Bring files together

################################################################################  
    
steps_40 = open('40_steps.txt', 'r').read().split()
intensity = [(int(i)) for i in steps_40]
steps_37 = np.array(intensity[0:37])

peak_val1 = full_peak_val[0]
peak_pos1 = full_peak_pos[0]
error_1 = full_errors[0]

peak_val2 = full_peak_val[1]
peak_pos2 = full_peak_pos[1]
error_2 = full_errors[1]

peak_val3 = full_peak_val[2]
peak_pos3 = full_peak_pos[2]
error_3 = full_errors[2]


# Different now, defie also pixel you want
peak_val = np.concatenate([peak_val1 ,peak_val2, peak_val3] ,axis = 1)
peak_pos = np.concatenate([peak_pos1 ,peak_pos2, peak_pos3] ,axis = 1)
error_ = np.concatenate([error_1 ,error_2, error_3] ,axis = 1)
error_sys = np.ones(np.shape(error_)) * 10 # Add a systematicc error of 5mA

error_compl = error_ + error_sys
int1 = full_number_bins[0]
int2 = full_number_bins[1] + int1
int3 = full_number_bins[2] + int2    

peak_val[1212] = 0
peak_val[1185] = 0
peak_val[255] = 0

########################## 25 represents where temp dep shows up ############################################################

# Fit through all good pixels
def func(x,a,b):
    return(a * x + b)

def func2(x,c,d):
    return(c*x**d)

def full_fit(x,a,b,c,d):
    return(a * x + b + c*x**d)



startingp = []
good_pixel = []
for pix in range(2048):
    for l in range(37):
        if peak_val[pix][l] > 0:            
            startingp.append(l)
            good_pixel.append(pix)
            break
 
startingp = np.array(startingp)
good_pixel = np.array(good_pixel)


popt_pix = []  
chisq = []
p_val = []
abs_err = []  
for pix in range(len(good_pixel)):
    
    region_fit = range(startingp[pix],25)
    popt, pcov  = curve_fit(func, steps_37[region_fit],peak_val[good_pixel[pix]][region_fit], p0=(1, 1),sigma = error_compl[good_pixel[pix]][region_fit],absolute_sigma=False) 
    popt_pix.append(popt)
    
    cond_1 = np.where(func(steps_37,*popt) > 0 )[0]
    cond_2 = np.where(peak_val[good_pixel[pix]] > 0 )[0]
    cond_3 = region_fit
    cond_u = list(set(cond_1).intersection(cond_2))
    cond_u = list(set(cond_u).intersection(cond_3))
    Ndf = len(cond_u) - 2

    chisq.append(chisquare(peak_val[good_pixel[pix]][cond_u],func(steps_37,*popt)[cond_u])[0])
#    print(chisquare(peak_val[good_pixel[pix]][cond_u],func(steps_37,*popt)[cond_u])[0]/Ndf)
    
    p_val.append(chisquare(peak_val[good_pixel[pix]],func(steps_37,*popt_pix[pix]))[1])
    abs_err.append((peak_val[good_pixel[pix]] - func(steps_37,*popt_pix[pix])))





###################################################################################
  
pix = 23

######################## Real Double Plot #########################################



def func2(x,c):
    return(c*x**2)
    
def full_fit(x,a,b,c):
    return(a * x + b + c*x**2)
    

def new_fit(x,a,b,c,d,e):
    return(a*x**4 + b*x**3 + c*x**2 + d*x + e)
    

def new_fit(x,a,b,c):
#    return(a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f)
    return(a * x**2 + b*x + c)
  
coeff_pix = []
for pix in range(len(good_pixel)):

    y_axis = np.zeros(steps_37.shape)
    y_axis[26::1] = abs(abs_err[pix][26::1])
    y_axis = abs(abs_err[pix])
    sigma_ = error_compl[good_pixel[pix]]
    popt1, pcov  = curve_fit(func2, steps_37,y_axis, p0=(1),sigma = sigma_,absolute_sigma=False) 
    coeff_pix.append(popt1)

coeff_pix = np.array(coeff_pix)



full_func = []
full_popt = []

for pix in range(len(good_pixel)):
    sigma_ = error_compl[good_pixel[pix]]
    full_func.append(func(steps_37,*popt_pix[pix]) - func2(steps_37,*coeff_pix[pix]))
    # Real fit function parameters
    popt, pcov  = curve_fit(new_fit, steps_37,full_func[pix], p0=(1,1,1),sigma = sigma_ ,absolute_sigma=False)
    full_popt.append(popt) 

full_func = np.array(full_func)
full_popt = np.array(full_popt)

##################################for one pixel ##########################################

pix = 23

# Get real fit residuals
sigma_ = error_compl[good_pixel[pix]] 
residuals = peak_val[good_pixel[pix]] - new_fit(steps_37,*full_popt[pix]) 
residual_syserr = residuals * 0.05
rel_residual = residuals/peak_val[good_pixel[pix]] 


sigma_Q = abs(sigma_) + abs(residual_syserr)  # sdt + 5 mA abs_sys + rel 3% 2 times
sigma_Q_rel = sigma_Q / peak_val[good_pixel[pix]] 


sig_ = abs(1/peak_val[good_pixel[pix]])*(abs(sigma_) + abs(residual_syserr)  + abs(residuals) * 0.05) + abs((residuals/(peak_val[good_pixel[pix]]**2)))*sigma_Q

sig_1 = error_[pix] / peak_val[good_pixel[pix]] 
sig_2 = abs(1 / peak_val[good_pixel[pix]] * (error_compl[pix])) + sig_1
sig_3 = sig_2 + residual_syserr/ peak_val[good_pixel[pix]] 

plt.figure(figsize = (12,8))
#plt.title("Fit of Pixel " + str(pix))
plt.plot(steps_37,func(steps_37,*popt_pix[pix]), c="cornflowerblue") # Linear fit
plt.errorbar(steps_37,peak_val[good_pixel[pix]],yerr = sigma_Q ,capsize = 4 ,elinewidth = 1,fmt = ".", c="orange") # True points
plt.plot(steps_37,new_fit(steps_37,*full_popt[pix]),"-.", c="limegreen") # real fit function
plt.xlabel("Intensity / MHz",size = 20)
plt.ylabel("Amplitude / mV",size = 20)
plt.legend(["Linear Fit","Quadratic Fit","Data"],prop={'size': 20})
#for r in ([int1,int2]):
#    plt.axvline(steps_37[r],c="red",ls="--")
#plt.axvline(steps_37[-1],c="red",ls="--") 


plt.title("Comparison before and after of Pixel " + str(pix))
for r in ([int1,int2]):
    plt.axvline(steps_37[r],c="red",ls="--")
plt.axvline(steps_37[-1],c="red",ls="--") 
#plt.plot(steps_37[startingp[pix]::1],abs_err[pix][startingp[pix]::1],"*")
plt.errorbar(steps_37[startingp[pix]::1],abs_err[pix][startingp[pix]::1],yerr= sigma_[startingp[pix]::1] ,capsize = 4 ,elinewidth = 1,fmt = ".")
plt.xlabel("Intens steps")
plt.ylabel("absolut residual")




plt.figure(figsize = (15,9)) #plt.plot(steps_37[startingp[pix]::1],residual[startingp[pix]::1],"*")
plt.errorbar(steps_37[startingp[pix]::1],residuals[startingp[pix]::1],yerr= residual_syserr[startingp[pix]::1] ,capsize = 4 ,elinewidth = 1,fmt = ".")
plt.xlabel("Intensity / MHz",size = 15)
plt.ylabel("Absolute residuals",size = 15)
plt.axhline(0,c="k",ls="--") 




plt.figure(figsize = (10,8))
#plt.plot(steps_37[startingp[pix]::1],rel_residual[startingp[pix]::1],"*")
plt.xlabel("Intensity / MHz",size = 20)
plt.ylabel("Relative Residuals",size = 20)
plt.semilogx()
plt.axhline(0,c="k",ls="--") 
plt.errorbar(steps_37[startingp[pix]::1],rel_residual[startingp[pix]::1],yerr=  sig_3 ,capsize = 4 ,elinewidth = 1,fmt = ".", c = "r")
plt.errorbar(steps_37[startingp[pix]::1],rel_residual[startingp[pix]::1],yerr=  sig_2 ,capsize = 4 ,elinewidth = 1,fmt = "none", c = "b")
plt.errorbar(steps_37[startingp[pix]::1],rel_residual[startingp[pix]::1],yerr=  sig_1,capsize = 4 ,elinewidth = 1,fmt = "none", c = "green")
#plt.legend(["",'$ /sigma_{tot}$','$ /sigma_{sys}^{abs}$','$ /sigma_{sys}^{rel}$'], prop={'size': 20})
plt.ylim([-2,2])
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

#plt.rc('xtick',labelsize=10)
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=10)
#plt.errorbar(steps_37[startingp[pix]::1],rel_residual[startingp[pix]::1],yerr= sig_[startingp[pix]::1] ,capsize = 4 ,elinewidth = 1,fmt = ".")



cond_1 = np.where(new_fit(steps_37,*full_popt[pix]) > 0 )[0]
cond_2 = np.where(peak_val[good_pixel[pix]] > 0 )[0]
cond_u = list(set(cond_1).intersection(cond_2))
Ndf = len(cond_u) - 3
print(chisquare(peak_val[good_pixel[pix]][cond_u],new_fit(steps_37,*full_popt[pix])[cond_u])[0] / Ndf)
#print(sum((peak_val[good_pixel[pix]][cond_2] - full_fit(steps_37,*full_popt[pix])[cond_2])**2/ full_fit(steps_37,*full_popt[pix])[cond_2]))

##################################### all the pixels ###########################################

chi2_all = []
for pix in range(len(good_pixel)):

    cond_1 = np.where(new_fit(steps_37,*full_popt[pix]) > 0 )[0]
    cond_2 = np.where(peak_val[good_pixel[pix]] > 0 )[0]
    cond_u = list(set(cond_1).intersection(cond_2))
    Ndf = len(cond_u) - 3
    chi2_all.append(chisquare(peak_val[good_pixel[pix]][cond_u],new_fit(steps_37,*full_popt[pix])[cond_u])[0] / Ndf)

    
#    print(chisquare(peak_val[good_pixel[pix]][cond_2],full_fit(steps_37,*full_popt[pix])[cond_2])[0])
#    print(sum((peak_val[good_pixel[pix]][cond_2] - full_fit(steps_37,*full_popt[pix])[cond_2])**2/ full_fit(steps_37,*full_popt[pix])[cond_2]))

print(np.mean(chi2_all))



















