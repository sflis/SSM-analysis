#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:52:22 2019

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

from ssm.core import pchain
from ssm.core.util_pmodules import Aggregate
from ssm.pmodules import *

steps_40 = [23,26,30,34,39,45,52,59,68,78,89,102,117,134,153,176,201,231,264,303,347,397,455,521,597,684,784,898,1028,1178,1349,1545,1770,2028,2323,2661,3048,3492,4000]
number_pixels = 2048
bad_pixel = np.array([25, 58, 96,97,98,99, 101, 226, 247, 256, 259, 304, 448, 449,570, 653, 670, 776, 1049, 1094,
                      1158, 1177,1185, 1212, 1352, 1367, 1381,1427, 1434,1439,1503, 1562, 1680, 1765,
                      1812, 1813, 1814, 1815, 1829, 1852, 1869, 1945, 1957, 2009, 2043])
current_path = "/data/CHECS-data/nsbcal/"

folder = []
directory = '/data/CHECS-data/nsbcal/'
for filename in os.listdir(directory):
    if filename.endswith(".hdf5"):
        folder.append(filename)    
folder


#40steps_NSB-Cal40_2019-04-26_11.08
#40steps_NSB-Cal40_2019-04-26_11.20
#40steps_NSB-Cal40_2019-04-26_11.28
file = folder[12]

# initializing our process chain
data_proc = pchain.ProcessingChain()

reader = Reader(directory+file)
data_proc.add(reader)
#frame_cleaner = PFCleaner()
#The Aggregate module collects the computed object from the frame
aggr = Aggregate(["raw_resp"])
data_proc.add(aggr)
print(data_proc)
data_proc.run()
data = aggr.aggr["raw_resp"][0]

get_data = np.nan_to_num(data.data)
get_data[get_data > 4000] = 0
dt = data.time - data.time[0]

time_mask = np.zeros(data.data[:,23].shape,"bool")
time_mask[np.where(get_data[:,23] > 0)] = True
peak_int = get_data[time_mask,:]


index_mask = np.where(time_mask == True)[0]

index = []
index.append(index_mask[0])
for l in range(len(index_mask)-1):
    if index_mask[l+1] > index_mask[l] + 400:
        index.append(index_mask[l+1])
int_begin = np.array(index)

index_mask = np.concatenate((index_mask,([index_mask[-1] + 500])))
index = []
for l in range(len(index_mask)-1):
    if index_mask[l+1] > index_mask[l] + 400:
        index.append(index_mask[l])
int_end = np.array(index)
interval_start_stop = np.sort(np.concatenate((int_begin,int_end)))

arr = []
for l in range(len(interval_start_stop)-1):
    arr.append(np.arange(interval_start_stop[l],interval_start_stop[l+1]))
time_int = arr[::2]


pix =  23
peak_pos =[]
peak_val = []
for pix in range(2048):
    peaks_pos = []
    peaks_val = []
    for t in range(len(time_int)):
        if np.any(find_peaks(get_data[time_int[t],pix])[0]):
            peaks_pos.append(time_int[t][find_peaks(get_data[time_int[t],pix])[0]])      
            peaks_val.append(get_data[time_int[t][find_peaks(get_data[time_int[t],pix])[0]],pix])
        
    peak_pos.append(peaks_pos)
    peak_val.append(peaks_val)
peak_pos = np.asarray(peak_pos)
peak_val = np.asarray(peak_val)


peak_all_pixel = []
pos_all_pixel = []
error_stat = []
for pix in range(2048):
    peak_ = []
    pos_ = []
    error_ = []
    for l in range(len(peak_val[pix])):
        if len(peak_val[pix][l])>= 5:
            peak_.append(np.median(peak_val[pix][l][0:5]))
            pos_.append(peak_pos[pix][l][np.where(peak_val[pix][l][0:5]== peak_[l])[0][0]])
            error_.append(np.std((peak_val[pix][l][0:5])))
        else:
            peak_.append(np.amax(peak_val[pix][l][0:len(peak_val[pix][l])]))
            pos_.append(peak_pos[pix][l][np.where(peak_val[pix][l][0:5]== peak_[l])[0][0]])
            error_.append(np.std(np.amax(peak_val[pix][l][0:len(peak_val[pix][l])])))
    peak_all_pixel.append(peak_)
    pos_all_pixel.append(pos_)
    error_stat.append(error_)
#peak_all_pixel = np.array(peak_all_pixel)  
#pos_all_pixel = np.array(pos_all_pixel)  
#error_stat = np.array(error_stat)


property_path = os.path.join(os.getcwd()+'/', 'output_40steps_'+file[0:-5])
peak_porperties = {"Peak_val":peak_all_pixel, "Position_peaks":pos_all_pixel, "Interval_start+stop":interval_start_stop, "Good_pixel":time_mask, "Std_Error":error_stat}

with open(property_path , 'wb') as handle:
    pickle.dump(peak_porperties, handle)

with open(property_path , 'rb') as handle:
    readout = pickle.load(handle)

#########################################################################

plt.figure() 
for pix in range(21,24):   
    plt.plot(pos_all_pixel[pix],peak_all_pixel[pix],"*")


plt.figure()
plt.plot(peak_pos[23],peak_val[23],"*")    


plt.figure()
plt.plot(dt[time_mask],get_data[time_mask,21],"*")
for l in int_begin:
    plt.axvline(dt[l],c="k")
for k in int_end:
    plt.axvline(dt[k],c="k")

plt.plot(find_peaks(get_data[time_mask,23])[0],"+")
##########################################################################

full_inter = []
full_peak_pos = []
full_peak_val = []
full_errors = []
full_number_bins = []
for file in range(10,13):

#    # Choose file in folder
    file = folder[file]
#    property_path = os.path.join('/home/wolkerst/projects/cta/SSM-analysis/','40steps_'+file[0:-5])
    property_path = os.path.join(os.getcwd()+'/', 'output_40steps_'+file[0:-5])
    with open(property_path, 'rb') as handle:
        get_saved_file = pickle.load(handle)
#       {"Peak_val":peak_all_pixel, "Position_peaks":pos_all_pixel, "Interval_start+stop":interval_start_stop, "Good_pixel":time_mask} 
    
    pxl = get_saved_file['Good_pixel']
    inter = get_saved_file['Interval_start+stop']
    inter_ext = get_saved_file["Interval_start+stop"]
    peak_pos = get_saved_file['Position_peaks']
    peak_val = get_saved_file['Peak_val']
    number_bins = int(len(inter)/2)
    error = get_saved_file['Std_Error']


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

"""
#Hi samuel
Sorry that i couldnt get everything back,
this file gives you the peaks and this time I it does 
not take the median of the five highest points but the median 
of the first 5 peaks. Also it only has the points which were greter than zero, so the array may be shoerter than 37,
but thus you dont have to ignore the first points, which otherwise would be zero, when fitting.

If you have any questions about any code, just let me know!

Se you!



#









