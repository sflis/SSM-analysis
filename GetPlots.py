#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 00:30:58 2019

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

folder = []
directory = '/data/CHECS-data/nsbcal/'
for filename in os.listdir(directory):
    if filename.endswith(".hdf5"):
        folder.append(filename)    
folder

from ssm.core import pchain
from ssm.core.util_pmodules import Aggregate
from ssm.pmodules import *

# initializing our process chain
data_proc = pchain.ProcessingChain()

# Choose file in folder
file = folder[10]
reader = Reader(directory+file)
# reader = Reader('/data/CHECS-data/nsbcal/NSB-Cal2019-04-26_08.57.hdf5')
data_proc.add(reader)
aggr = Aggregate(["raw_resp"])
data_proc.add(aggr)
data_proc.run()
pxl = 23

from CHECLabPy.plotting.camera import CameraImage
data = aggr.aggr["raw_resp"][0]

dt = data.time - data.time[0]
print(dt)
number_pixels = 2048


# Remove bad pixels
bad_pixel = np.array([25, 58, 101, 226, 256, 259, 304, 448, 449,570, 653, 670, 776, 1049, 1094,
                      1158, 1177, 1352, 1367, 1381,1427, 1434,1439,1503, 1562, 1680, 1765,
                      1812, 1813, 1814, 1815, 1829, 1852, 1869, 1945, 1957, 2009, 2043])

clean_data = data.data.copy()

clean_data[:,bad_pixel[:]] = 0
clean_data = np.nan_to_num(clean_data)
start = time.time()

# Remove values above 4000 to still obtain (valuable) info of a pixel with maybe some distubance
clean_data[np.where(clean_data[:,:] > 4000),:] = 0

end = time.time()
print("Time for removal: "+ str(end - start))



start = time.time()

interval_peaks = find_peaks(clean_data[:,pxl])[0]

inter = []
inter.append(interval_peaks[0])
for number in range(len(interval_peaks)-1) :
    
    if clean_data[interval_peaks[number+1],pxl]-clean_data[interval_peaks[number],pxl] > 1.5:
        inter.append(interval_peaks[number])
        inter.append(interval_peaks[number +1])
inter.append(interval_peaks[-1])

interval_ext = np.asarray(inter)
# interval_ext = interval_ext 
# interval_ext[::2] = interval_ext[::2] 

interval_ext = interval_ext + 20
interval_ext[::2] = interval_ext[::2] - 10

# Defining extended intervals
number_bins = int(len(inter)/2)
# Filling up the interval
dummy_arr = range(2*number_bins)[::2]
int_compl = []
for i in dummy_arr:
    int_compl.append(np.arange(interval_ext[i],interval_ext[i+1]))
end = time.time()
print(end-start)
interval = np.concatenate(int_compl)
# # print(int_compl[0])

print("Intervals: " +str(inter))
print("Interval after expansion: " + str(interval_ext))
print("Number of bins: " + str(number_bins))
print(interval_ext[::2])


property_path = os.path.join('/home/wolkerst/projects/cta/SSM-analysis/output_datafiles', 'output_')

with open(property_path+file[0:-5], 'rb') as handle:
    b = pickle.load(handle)



pxl = b['Good_pixel']
inter = b['Interval_begin+end']
inter_ext = b["Extended_interval_begin+end"]
peak_pos = b['Position_of_Peaks_in_time']
peak_val = b['Value_of_Peaks']
number_bins = int(len(inter)/2)
# check for additional note
# look at data again 
plt.figure()
cond_1 = np.where(clean_data[:,23] > 0)
plt.plot(dt[cond_1],clean_data[:,pxl][cond_1],"o")
# for l in range(10,20):
#     plt.plot(dt,clean_data[:,l],"o")
plt.plot(peak_pos[pxl],peak_val[pxl],"*")
#plt.plot(dt,clean_data[:,pxl],"o")


for l in range(len(inter)):
    plt.axvline(inter_ext[l],c="red",ls="--")
#plt.title(str(file))
plt.xlabel('Time / s')
plt.legend(["data points", "selected peaks","selected interval"], loc=0)
plt.ylabel("Amplitude / mHz")
plt.axhline(0,c="grey",ls="-")

print("Intervals: " +str(inter))
print("Interval after expansion: " + str(inter_ext))
print("Number of bins: " + str(number_bins))
