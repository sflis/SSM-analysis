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

import dashi
dashi.visual()

runfile = "Run13646_ss"

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
                    
zmin_intspace = readout["zmin_intspace_"]
zmax_intspace = readout["zmax_intspace_"]
zmin_caldat = readout["zmin_caldat_"]
zmax_caldat = readout[ "zmax_caldat_"]
zmin_offset = readout["zmin_offset"]
zmax_offset = readout["zmax_offset"]





# Visualize for some pixels:
plt.figure()   
plt.axvline(dt[int_begin],color = "k")
plt.axvline(dt[int_end],color = "k") 
plt.xlabel("Time / s")
plt.ylabel("Amplitude / (mV)")
plt.title("Interval of offset")
for i in [1,23,600,900,1200]:
    plt.plot(dt,calibrated_data[:,i])   
#plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_std_interval"))


#A time series plot
plt.figure()
plt.plot(data.time-data.time[0],int_space_averaged)
plt.xlabel('Time since run start (s)')
plt.ylabel("Average amplitude (mV)")
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
camera.image = calibrated_data[0,:]
camera.add_colorbar('Intensity / MHz')
zmin_calbdat = min(int_space_averaged) - 0.05*min(int_space_averaged) 
#zmax_calbdat = 235
zmax_calbdat = max(int_space_averaged) + 0.05*max(int_space_averaged) -10
camera.set_limits_minmax(zmin=zmin_calbdat,zmax = zmax_calbdat)
camera.ax.set_title('Calibrated Data')
#plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_calibrated_data"))

camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = c_
camera.add_colorbar('Intensity / MHz')
camera.ax.set_title('Flat field coefficents $c_{i}$')
#plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_flat_field_coeffs_c"))

camera = CameraImage(data.xpix, data.ypix, data.size)
camera.image = offset_calibrated_data[0,:]
camera.add_colorbar('Intensity / MHz')
camera.ax.set_title('Offset of calibrated data')
zmin_offset = None
zmax_offset = None
#np.where(offset_calibrated_data > 140)
zmin_offset = 0
#zmax_offset = 60
camera.set_limits_minmax(zmin=zmin_offset,zmax = zmax_offset)
plt.savefig(os.path.join(current_path+'/'+ runfile, runfile+"_offset_calibrated_data"))
