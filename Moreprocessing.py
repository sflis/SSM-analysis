#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:34:01 2019

@author: wolkerst

"""
#cd projects/cta/SSM-analysis
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

import dashi
dashi.visual()

#def extract(lst):
#    return(lst[i][0] for i in range(len(lst)))
#    
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
# # Read in raw data
# =============================================================================
# Pick runfile
runfile = "Run13637_ss"

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
#############################################################################

# Remove eventally new broken pixels
new_good_pix = np.ones(data.data.shape,dtype = "bool")
new_good_pix[np.nan_to_num(data.data) == 0] = False
new_good_pix[:,1404] = False
new_good_pix[:,1426] = False
new_good_pix[:,1755] = False

def remback_int(calibrated_data, dt, new_good_pixel):
    # Get stable interval from minimum of std, from calibrated data in 4 min measurement intervals
    min_of_run = math.floor(dt[-1]/60)
#    min_of_run = 1
    number_of_intervals = int(min_of_run / 3)
    
    #Splits calibrated data into 4 min intervals
    cal_interval_ = np.array_split(calibrated_data,number_of_intervals,axis= 0)
    cal_interval_std = [np.std(cal_interval_[:][i], axis = 0) for i in range(number_of_intervals)]
    cal_std = [np.nanmean(cal_interval_std[i][new_good_pix[0,:] == 1], axis = 0) for i in range(number_of_intervals)]
    
    bin_of_min_std = np.where(cal_std == min(cal_std))[0][0]
    min_cal_std = np.array_split(dt, number_of_intervals)[bin_of_min_std]
    min_cal_std = np.array_split(dt, number_of_intervals)[-2]
    
    int_start = list(dt).index(min_cal_std[0])
    int_stop  = list(dt).index(min_cal_std[-1])
    
    return(int_start, int_stop)

# Get intensity
intensity = invf(data.data,a,b,d)
intensity[new_good_pix == 0] = -1 # Set bad pix to dummy intensity unequal zero

# Get time interval with stable background:
int_begin, int_end = remback_int(data.data,dt,new_good_pix)

int_time_averaged = np.mean(intensity[[int_begin,int_end],:], axis = 0) # Mean of each pixel individually over stable int time
int_space_averaged = np.mean(intensity[:,new_good_pix[0][:]], axis = 1) # mean of all pixels at each time frame
int_spacetime_averaged = np.mean(int_space_averaged)

# Coefficient for faltfieding: I_cal = c_ * I__raw_data_after_inversion
c_ = int_spacetime_averaged / int_time_averaged
# Reset the bad pix to zero
c_[new_good_pix[0,:] == 0] = 0
calibrated_data = c_ * intensity

## Get time interval with stable background:
#int_begin, int_end = remback_int(data.data,dt,new_good_pix)

# Get the offet of Calibrated data via the mean in a good interval
# I_offset_of_calibration = I_cal - offset_interval
off_int = np.mean(calibrated_data[int_begin:int_end], axis = 0)
offset_calibrated_data = calibrated_data - np.mean(calibrated_data[int_begin:int_end], axis = 0)

plt.figure()   
plt.axvline(dt[int_begin],color = "k")
plt.axvline(dt[int_end],color = "k") 
plt.xlabel("Time / s")
plt.ylabel("Amplitude / (mV)")
plt.title("Interval of offset")
for i in [1,23,600,900,1200]:
    plt.plot(dt,data.data[:,i])  
# =============================================================================
# # Save figures of calibration
# =============================================================================

import glob
read_from_path = os.path.join(os.getcwd(),path)
if not os.path.isdir(read_from_path[-16:-5]):
    os.mkdir(read_from_path[-16:-5])

#if not os.path.isdir(read_from_path[-13:-5]):
#    os.mkdir(read_from_path[-13:-5])

files_in_imagefolder = glob.glob(os.path.join(read_from_path,"*"))
for f in files_in_imagefolder:
    os.remove(f)

current_path = '/home/wolkerst/projects/cta/SSM-analysis'


# Visualize for some pixels:
plt.figure()   
plt.axvline(dt[int_begin],color = "k")
plt.axvline(dt[int_end],color = "k") 
plt.xlabel("Time / s")
plt.ylabel("Amplitude / (mV)")
plt.title("Interval of offset")
for i in [1,23,600,900,1200]:
    plt.plot(dt,calibrated_data[:,i])   
plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_std_interval"))


#A time series plot
plt.figure()
plt.plot(data.time-data.time[0],int_space_averaged)
plt.xlabel('Time since run start (s)')
plt.ylabel("Average amplitude (mV)")
plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_space_averaged_over_time"))

#Different average camera images
camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = int_time_averaged
zmin_intspace = min(int_space_averaged) - 0.05*min(int_space_averaged)
zmax_intspace = max(int_space_averaged) + 0.05*max(int_space_averaged)
camera.set_limits_minmax(zmin=zmin_intspace,zmax = zmax_intspace)
camera.add_colorbar('Amplitdue (mV)')
camera.ax.set_title('Time averaged data')
plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_camera_time_averaged"))

camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = calibrated_data[0,:]
camera.add_colorbar('Amplitdue (mV)')
zmin_calbdat = min(int_space_averaged) - 0.01*min(int_space_averaged) 
#zmin_calbdat = 380
zmax_calbdat = max(int_space_averaged) + 0.01*max(int_space_averaged) 
camera.set_limits_minmax(zmin=zmin_calbdat,zmax = zmax_calbdat)
camera.ax.set_title('Calibrated Data')
plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_calibrated_data"))

camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = c_
camera.add_colorbar('Amplitdue (mV)')
camera.ax.set_title('Flat field coefficents $c_{i}$')
plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_flat_field_coeffs_c"))

camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = offset_calibrated_data[0,:]
camera.add_colorbar('Amplitdue (mV)')
camera.ax.set_title('Offset of calibrated data')
zmin_offset = None
zmax_offset = None
#np.where(offset_calibrated_data > 50)
#zmin_offset = 0
#zmax_offset = 60
camera.set_limits_minmax(zmin=zmin_offset,zmax = zmax_offset)
plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_offset_calibrated_data"))


# =============================================================================
# # Save properties of calibration to images:
# =============================================================================

property_path = os.path.join(current_path+'/'+ runfile, "calibration_properties")
calibration_properties = {"Interval_offset":(int_begin,int_end),"New_good_pixel":new_good_pix,"Time_averaged_int":int_time_averaged, "Space_averaged":int_space_averaged, 
                          "Calibrated_data":calibrated_data, "ff_coefficients_c":c_, "Offset_calibrated_data":offset_calibrated_data,                    
                          "zmin_intspace_":zmin_intspace,"zmax_intspace_":zmax_intspace, "zmin_caldat_":zmin_calbdat, "zmax_caldat_":zmax_calbdat,
                          "zmin_offset":zmin_offset,"zmax_offset":zmax_offset}
with open(property_path , 'wb') as handle:
    pickle.dump(calibration_properties, handle)


###################################################################################################


property_path = os.path.join(current_path+'/'+ "Run13730_ss", "calibration_properties")
with open(property_path , 'rb') as handle:
    dummy1 = pickle.load(handle)

# Pick runfile
runfile = "Run13730_ss"

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

coeff_13730 = dummy1["ff_coefficients_c"]

camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = coeff_13730
camera.add_colorbar('Amplitdue (mV)')
camera.set_limits_minmax(zmin=0,zmax = 1.6)
camera.ax.set_title('Flat field coefficents $c_{i}$ ' + str(runfile))

###################################################################################################

property_path = os.path.join(current_path+'/'+ "Run13638_ss", "calibration_properties")
with open(property_path , 'rb') as handle:
    dummy2 = pickle.load(handle)
    
    
coeff_13638 = dummy2["ff_coefficients_c"]

runfile = "Run13638_ss"

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


camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = coeff_13638
camera.add_colorbar('Amplitdue (mV)')
camera.set_limits_minmax(zmin=0,zmax = 1.6)

camera.ax.set_title('Flat field coefficents $c_{i}$  ' + str(runfile))

###################################################################################################


property_path = os.path.join(current_path+'/'+ "Run13637_ss", "calibration_properties")
with open(property_path , 'rb') as handle:
    dummy3 = pickle.load(handle)
    
    
coeff_13637 = dummy3["ff_coefficients_c"]

runfile = "Run13637_ss"

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


camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = coeff_13637
camera.add_colorbar('Amplitdue (mV)')
camera.set_limits_minmax(zmin=0,zmax = 1.6)

camera.ax.set_title('Flat field coefficents $c_{i}$  ' + str(runfile))


###################################################################################################

rc2 = np.nan_to_num((coeff_13637 - coeff_13638) / coeff_13638)

plt.figure(figsize = (8,6))
plt.hist(rc2, bins = 100, range=(-0.1,0.1))
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend([r"$\frac{c_{Run13637} - c_{Run13638}}{c_{Run13638}}$ "] , fontsize = 19, loc = 1)
plt.axvline(0,c="k")

rc = np.nan_to_num((coeff_13638 - coeff_13730) / coeff_13730)

plt.figure(figsize = (8,6))
plt.hist(rc, bins = 100, range=(-0.2,0.2))
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.legend([r"$\frac{c_{Run13638} - c_{Run13730}}{c_{Run13730}}$ "] , fontsize = 20, loc = 0)
plt.axvline(0,c="k")



