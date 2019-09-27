#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:31:24 2019

@author: wolkerst
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
import time
import pickle
import os
from scipy.signal import find_peaks, savgol_filter

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import math
from tqdm import tqdm
from ssm.core import pchain
from ssm.pmodules import *
from ssm.core.util_pmodules import Aggregate,SimpleInjector
from ssm.pmodules import *

from ssm.pmodules import Reader
from CHECLabPy.plotting.camera import CameraImage
from pylab import *
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from operator import itemgetter
from ssm.core import pchain


def invf(y,a,b,c):
    
    return np.nan_to_num(-(b-np.sqrt(b*b -4*a*(c-y)))/(2*a))

current_path = '/home/wolkerst/projects/cta/SSM-analysis'
property_path = os.path.join(current_path+'/', "fit_parameters")

# =============================================================================
# # Load fit parameters
# =============================================================================
with open(property_path , 'rb') as handle:
    props = pickle.load(handle)
print(props)

# coefficient of f(x) = ax^2 + bx + d
a = props["all_parameters"][:,0]
b = props["all_parameters"][:,1]
d = props["all_parameters"][:,2]

# =============================================================================
# #Read in data and saved file
# =============================================================================

runfile = "Run13771_ss"

data_proc = pchain.ProcessingChain()
path = '/data/CHECS-data/astri_onsky/slowsignal/'+str(runfile)+'.hdf5'
reader = Reader(path)
data_proc.add(reader)
#This module removes incomplete frames and marks bad and unstable pixels
frame_cleaner = PFCleaner()
data_proc.add(frame_cleaner)
#The Aggregate module collects the computed object from the frame
aggr = Aggregate(["raw_resp"])
data_proc.add(aggr)
# Simple visualization of the chain
print(data_proc)
#Execute the chain
data_proc.run()

data = aggr.aggr["raw_resp"][0]
dt = data.time - data.time[0]

current_path = '/home/wolkerst/projects/cta/SSM-analysis'
property_path = os.path.join(current_path+'/'+ runfile, "calibration_properties")
#calibration_properties = {"Interval_offset":(int_begin,int_end),"New_good_pixel":new_good_pix,"Time_averaged_int":int_time_averaged, "Space_averaged":int_space_averaged, 
#                          "Calibrated_data":calibrated_data, "ff_coefficients_c":c_, "Offset_calibrated_data":offset_calibrated_data,                    
#                          "zmin_intspace_":zmin_intspace,"zmax_intspace_":zmax_intspace, "zmin_caldat_":zmin_calbdat, "zmax_caldat_":zmax_calbdat,
#                          "zmin_offset":zmin_offset,"zmax_offset":zmax_offset}

with open(property_path , 'rb') as handle:
    readout = pickle.load(handle)
    
int_begin, int_end = readout["Interval_offset"]

int_time_averaged = readout["Time_averaged_int"]
int_space_averaged = readout["Space_averaged"]
calibrated_data = readout["Calibrated_data"]
c_ = readout["ff_coefficients_c"]
offset_calibrated_data = readout[ "Offset_calibrated_data"]
new_good_pix = readout["New_good_pixel"]                   
zmin_intspace = readout["zmin_intspace_"]
zmax_intspace = readout["zmax_intspace_"]
zmin_caldat = readout["zmin_caldat_"]
zmax_caldat = readout[ "zmax_caldat_"]
zmin_offset = readout["zmin_offset"]
zmax_offset = readout["zmax_offset"]


plt.figure()   
plt.axvline(dt[int_begin],color = "k")
plt.axvline(dt[int_end],color = "k") 
plt.xlabel("Time / s")
plt.ylabel("Amplitude / (mV)")
plt.title("Interval of offset")
for i in range(1000):
    plt.plot(dt,data.data[:,i])  

# =============================================================================
# 
# Load 13312
# 
# =============================================================================

# Pick runfile
runfile = "Run13312"



# initializing our process chain
data_proc = pchain.ProcessingChain()
path = '/data/CHECS-data/astri_onsky/slowsignal/'+str(runfile)+'.hdf5'
reader = Reader(path)
data_proc.add(reader)
#This module removes incomplete frames and marks bad and unstable pixels
frame_cleaner = PFCleaner()
data_proc.add(frame_cleaner)
#The Aggregate module collects the computed object from the frame
aggr = Aggregate(["raw_resp"])
data_proc.add(aggr)
# Simple visualization of the chain
print(data_proc)
#Execute the chain
data_proc.run()

data = aggr.aggr["raw_resp"][0]
dt = data.time - data.time[0]

# Intensity
new_good_pix = np.ones(data.data.shape,dtype = "bool")
new_good_pix[np.nan_to_num(data.data) == 0] = False
intensity = invf(data.data,a,b,d)
intensity[new_good_pix == 0] = 0 # Set bad pix to dummy intensity unequal zero


F_i = c_ * intensity
F_ioff = F_i - F_i[5000,:]

camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = F_i[0,:]
camera.add_colorbar('Intensity / MHz')
zmin_calbdat = min(int_space_averaged) - 0.01*min(int_space_averaged)
#zmax_calbdat = 235
zmax_calbdat = max(int_space_averaged) + 0.01*max(int_space_averaged)
#zmax_calbdat = 350
zmin_calbdat = 100
camera.set_limits_minmax(zmin=zmin_calbdat,zmax = zmax_calbdat)
camera.ax.set_title('Calibrated Data' + str(runfile))

def remback_int(calibrated_data, dt, new_good_pixel):
    # Get stable interval from minimum of std, from calibrated data in 4 min measurement intervals
    min_of_run = math.floor(dt[-1]/60)
#    min_of_run = 1
    number_of_intervals = int(min_of_run / 1)
    
    #Splits calibrated data into 4 min intervals
    cal_interval_ = np.array_split(calibrated_data,number_of_intervals,axis= 0)
    cal_interval_std = [np.std(cal_interval_[:][i], axis = 0) for i in range(number_of_intervals)]
    cal_std = [np.nanmean(cal_interval_std[i][new_good_pix[0,:] == 1], axis = 0) for i in range(number_of_intervals)]
    
    bin_of_min_std = np.where(cal_std == min(cal_std))[0][0]
    min_cal_std = np.array_split(dt, number_of_intervals)[bin_of_min_std]
    min_cal_std = np.array_split(dt, number_of_intervals)[10]
    
    int_start = list(dt).index(min_cal_std[0])
    int_stop  = list(dt).index(min_cal_std[-1])
    
    return(int_start, int_stop)



# Get time interval with stable background:
int_begin, int_end = remback_int(data.data,dt,new_good_pix)

int_time_averaged = np.mean(intensity[[int_begin,int_end],:], axis = 0) # Mean of each pixel individually over stable int time
int_space_averaged = np.mean(intensity[:,new_good_pix[0][:]], axis = 1) # mean of all pixels at each time frame
int_spacetime_averaged = np.mean(int_space_averaged)

property_path = os.path.join(current_path+'/'+ runfile, "CAL_with_"+str("Run13638")) #+str("Run13638")
CAL = {"Interval_offset":(int_begin,int_end),"New_good_pixel":new_good_pix,"Time_averaged_int":int_time_averaged, "Space_averaged":int_space_averaged, 
                          "Calibrated_data":F_i, "ff_coefficients_c":c_, "Offset_calibrated_data":F_ioff, "Int_inv":intensity,                   
                          "zmin_intspace_":zmin_intspace,"zmax_intspace_":zmax_intspace, "zmin_caldat_":zmin_calbdat, "zmax_caldat_":zmax_calbdat,
                          "zmin_offset":zmin_offset,"zmax_offset":zmax_offset}
with open(property_path , 'wb') as handle:
    pickle.dump(CAL, handle)
with open(property_path , 'rb') as handle:
    dummy = pickle.load(handle)

# =============================================================================
# # Plot and adjust the axes
# =============================================================================

int_begin, int_end = dummy["Interval_offset"]

# Visualize for some pixels:
plt.figure()   
plt.axvline(dt[int_begin],color = "k")
plt.axvline(dt[int_end],color = "k") 
plt.xlabel("Time / s")
plt.ylabel("Amplitude / (mV)")
plt.title("Interval of offset")
for i in [1,23,600,900,1200]:
    plt.plot(dt,data.data[:,i])   
plt.axvline(x=dt[5000])    
#plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_std_interval"))

#A time series plot
plt.figure()
plt.plot(data.time-data.time[0],int_space_averaged)
plt.xlabel('Time since run start (s)')
plt.ylabel("Average Intensity / MHz")
#plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_space_averaged_over_time"))

#Different average camera images
camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = int_time_averaged
zmin_intspace = min(int_space_averaged) - 0.05*min(int_space_averaged)
zmax_intspace = max(int_space_averaged) + 0.05*max(int_space_averaged)
camera.set_limits_minmax(zmin=zmin_intspace,zmax = zmax_intspace)
camera.add_colorbar('Intensity / MHz')
camera.ax.set_title('Time averaged data')
#plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_camera_time_averaged"))


camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = np.nan_to_num(F_i[0,:])
camera.add_colorbar('Amplitude / MHz')
zmin_calbdat = min(int_space_averaged) - 0.05*min(int_space_averaged) 
zmax_calbdat = max(int_space_averaged) + 0.05*max(int_space_averaged) -10

zmax_calbdat = 250
zmin_calbdat = 100
camera.set_limits_minmax(zmin=zmin_calbdat,zmax = zmax_calbdat)
camera.ax.set_title('Calibrated Data', fontsize = 16)



camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = np.nan_to_num(data.data[0,:])
camera.add_colorbar('Amplitude / mV')
zmin_calbdat = min(int_space_averaged) - 0.05*min(int_space_averaged) 
zmax_calbdat = max(int_space_averaged) + 0.05*max(int_space_averaged) -10

zmax_calbdat = 300
zmin_calbdat = 0
camera.set_limits_minmax(zmin=zmin_calbdat,zmax = zmax_calbdat)
camera.ax.set_title('Raw Data', fontsize = 16)


#plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_calibrated_data"))
camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = intensity[0,:]
camera.add_colorbar('Amplitude / MHz')
zmin_calbdat = min(int_space_averaged) - 0.05*min(int_space_averaged) 
zmax_calbdat = max(int_space_averaged) + 0.05*max(int_space_averaged) -10
zmax_calbdat = 250
zmin_calbdat = 100
camera.set_limits_minmax(zmin=zmin_calbdat,zmax = zmax_calbdat)
camera.ax.set_title('Raw Data', fontsize = 30)


camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = c_
camera.add_colorbar('')
camera.ax.set_title('Flat field coefficients', fontsize = 30)





o = a.copy()
p = b.copy()
r = d.copy()

o[new_good_pix[0,:] == 0]  = np.nan
p[new_good_pix[0,:] == 0]  = np.nan
r[new_good_pix[0,:] == 0]  = np.nan

camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = o
camera.add_colorbar('')
plt.tick_params(axis='both', which='minor', labelsize=30)
camera.ax.set_title('Fit coefficient a', fontsize=10)
camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = p
camera.add_colorbar('')
camera.ax.set_title('Fit coefficient b',fontsize=10)
camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = r
camera.add_colorbar('')
camera.ax.set_title('Fit coefficient c',fontsize=10)
#plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_flat_field_coeffs_c"))


plt.figure(figsize = (8,6))
plt.hist(o, bins = 100, range=(min(o)*0.999,max(o)*1.001))
plt.xticks(fontsize=10)
plt.yticks(fontsize=14)
plt.xlabel("a / quadratic dependency",fontsize=18)



plt.figure(figsize=(8,6))
plt.hist(p, bins = 100, range=(min(p)*0.9,max(p)*1.1))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("b / linear dependency",fontsize=18)



plt.figure(figsize=(8,6))
plt.hist(r, bins = 100, range=(min(r)*0.95,0.01))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("c / sensitivity cutoff",fontsize=18)

#plt.legend([r"$\frac{c_{Run133638} - c_{Run137030}}{c_{Run13730}}$ "] ,  loc = 0)


plt.tick_params(axis='both', which='minor', labelsize=30)



#camera = CameraImage(data.xpix, data.ypix, data.size)
#camera.image = offset_calibrated_data[0,:]
#camera.add_colorbar('Amplitdue (mV)')
#camera.ax.set_title('Offset of calibrated data')
#zmin_offset = None
#zmax_offset = None
##np.where(offset_calibrated_data > 140)
#zmin_offset = 0
##zmax_offset = 60
#camera.set_limits_minmax(zmin=zmin_offset,zmax = zmax_offset)
#plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_offset_calibrated_data"))
