#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 18:03:09 2019

@author: wolkerst
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 07:18:09 2019

@author: wolkerst
"""

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

import os
import subprocess
import glob


import dashi
dashi.visual()
from ssm import pmodules

def get_time(frame):
    
    m, s = divmod(frame, 60)
    h, m = divmod(m, 60)
    
    return(f"{int(h):02d}:{int(m):02d}:{int(s):02d}") 
# =============================================================================
# # Load in Data for Video
# =============================================================================
current_path = '/home/wolkerst/projects/cta/SSM-analysis'
#list_of_calibrations = 
# "'Run13769_ss','   
dumm = [ 'Run13638_ss' , 'Run13770_ss', "Run13768_ss" , 'Run13637_ss' ,
        'Run13646_ss' , 'Run13771_ss','Run13658_ss' , 'Run13772_ss', 'Run13666_ss' ,
        'Run13773_ss', 'Run13667_ss' , 'Run13777_ss', 'Run13668_ss' , 'Run13779_ss',
        'Run13669_ss', 'Run13681_ss' ,'Run13682_ss' ,'Run13691_ss' , 'Run13693_ss' , 
        'Run13721_ss'  , 'Run13722_ss' ,'Run13730_ss' , 'Run13731_ss', 'Run13733_ss',
        'Run13735_ss']

#dumm = ['Run13769_ss']
dumm = ["Run13637_ss"]
for runfile in dumm:
    print(runfile) 

    
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
    
    
    property_path = os.path.join(current_path+'/'+ runfile, "CAL_with_Run13638")
    
    
    with open(property_path , 'rb') as handle:
        readout = pickle.load(handle)
        
    int_begin, int_end = readout["Interval_offset"]
    
    int_time_averaged = readout["Time_averaged_int"]
    int_space_averaged = readout["Space_averaged"]
    calibrated_data = readout["Calibrated_data"]
    c_ = readout["Int_inv"]
    offset_calibrated_data = readout[ "Offset_calibrated_data"]
                        
    #zmin_intspace = readout["zmin_intspace_"]
    #zmax_intspace = readout["zmax_intspace_"]
    zmin_calbat = readout["zmin_caldat_"] 
#    zmin_calbat = 100
    zmax_calbat = readout[ "zmax_caldat_"]
#    zmax_calbat = max(c_[:,23])
    #zmin_offset = readout["zmin_offset"]
    #zmax_offset = readout["zmax_offset"]
    new_good_pix = readout["New_good_pixel"]
    ###########################################################################
    from ssm.pmodules import *
    from ssm import pmodules
    proc_chain = pchain.ProcessingChain()
    #Copy data so that we do not overwrite the original data
    ffdata = copy.deepcopy(data)
    # apply the flatfielding
    ffdata.data -= offset_calibrated_data
    ffdata.data[new_good_pix == 0] = 0
    
    ffdata.data = copy.deepcopy(calibrated_data)
    
    #The Simple injector just creates a frame with the content of the input dictionary
    injector =  SimpleInjector({'data':ffdata})
    proc_chain.add(injector)
    #Smoothing the signal maybe not so useful right now
    smooth = SmoothSlowSignal()
    proc_chain.add(smooth)
    #Finds hotspot clusters
    clust = pmodules.ClusterCleaning(1.2,0.9)
    clust.in_data = 'data'
    #smooth.out_data
    proc_chain.add(clust)
    
    #The Aggregate module collects the computed object from the frame
    #We want the clusters and the smooth data
    aggr = Aggregate(["clusters","smooth_data"])
    proc_chain.add(aggr)
    
    print(proc_chain)
    proc_chain.run()
    
    clusters = aggr.aggr['clusters'][0]
    #clusters
    smooth_data = aggr.aggr['smooth_data'][0]
    smooth_data.data[:,new_good_pix[0,:] == 0] = 0
    ########################################################################################
    
    if smooth_data.data.shape[0] < 2000:
        frame_n = 100
        range_frames = 20
    else:
        frame_n = 1000
        range_frames = 200
#    plt.set_cmap('Greys_r')
    
    
    #Different average camera images
    camera = CameraImage(smooth_data.xpix, smooth_data.ypix, smooth_data.size)
    im = copy.deepcopy(smooth_data.data[frame_n])
    im[np.isnan(im)] = np.nanmean(im)
    camera.image = im
    
    camera.add_colorbar('Amplitude / MHz')
    camera.highlight_pixels([item for sublist in clusters[frame_n] for item in sublist],color='r')
    #camera.set_limits_minmax(-20,100)
    
    
    #for each frame put all the cluster pixels in one list
    c = []
    for cc in clusters:
        c.append([item for sublist in cc for item in sublist])
    
    
    #We only make the movie with every 200th frame
    sel = range(0,len(smooth_data.data),range_frames)
    #make_ssmovie(smooth_data,sel,c, minmax = (0,250))
    read_from_path = os.path.join(os.getcwd(),path)
    
    
#    moviefolder = os.path.join('cal_movie_'+read_from_path[-16:-5])
#    if not os.path.isdir(moviefolder):
#        os.mkdir('cal_movie_'+read_from_path[-16:-5])
#        
        
    #Use only with Run13312 type
    moviefolder = os.path.join('CALI_3_'+runfile[3:8]+'_13638_'+read_from_path[-16:-5])
    if not os.path.isdir(moviefolder):
        os.mkdir('CALI_3_'+runfile[3:8]+'_13638_'+read_from_path[-16:-5])
    

    impath = os.path.join(current_path,moviefolder)
    files = glob.glob(os.path.join(impath,"*"))
    for f in files:
        os.remove(f)
    
    
    
    #minmax=(-10,100)
    red = sel
#    minmax = (min(data.data[:,0]) - 0.2*min(data.data[:,0]) ,max(data.data[:,0]) + 0.2*max(data.data[:,0]))
    minmax = (zmin_calbat, zmax_calbat)
    dpi=800
    scale=0.2
#    title=""
    fps=25
    filename="out"
    zlabel="Amplitude / mHz"
    highlightpix=c
    
    dpi -= int(scale*19.20*dpi)%2
    dpii =400
    scale = 0.70
    date = "2019-06-12 "
    time_00 = 76920
    fig,ax = plt.subplots(figsize=(1920/dpii*scale,1080/dpii*scale),dpi=dpii*scale)
    camera = CameraImage(data.xpix, data.ypix, data.size,ax=ax)
    
    camera.add_colorbar('Amplitude / MHz')
    camera.set_limits_minmax(*minmax)
    camera.ax.set_title(date+get_time(dt[0] + time_00))
    im = copy.deepcopy(calibrated_data[0])
    camera.image = im
    highl = None
    
  
    
    for i in tqdm(red,total=len(red)):
        im = copy.deepcopy(calibrated_data[i])
        im[np.isnan(im)] = np.nanmean(im)
        
        camera.ax.set_title(date+get_time(dt[i] + time_00))
        if highl is None:
            highl = camera.highlight_pixels(highlightpix[i])
        else:
            lw_array = np.zeros(camera.image.shape[0])
            lw_array[highlightpix[i]] = 0.5
            highl.set_linewidth(lw_array)
    
        camera.image = im
        plt.savefig(os.path.join(impath,"_calibrated_data%.10d.png"%i),dpi=dpi)
        
    
    plt.close("all")




current_path = '/home/wolkerst/projects/cta/SSM-analysis'
#list_of_calibrations = 
# "'Run13769_ss','   
#dumm = [ 'Run13638_ss' , 'Run13770_ss', "Run13768_ss" , 'Run13637_ss' ,
#        'Run13646_ss' , 'Run13771_ss','Run13658_ss' , 'Run13772_ss', 'Run13666_ss' ,
#        'Run13773_ss', 'Run13667_ss' , 'Run13777_ss', 'Run13668_ss' , 'Run13779_ss',
#        'Run13669_ss', 'Run13681_ss' ,'Run13682_ss' ,'Run13691_ss' , 'Run13693_ss' , 
#        'Run13721_ss'  , 'Run13722_ss' ,'Run13730_ss' , 'Run13731_ss', 'Run13733_ss',
#        'Run13735_ss']

#dumm = ['Run13769_ss']

import imageio


for readmov in dumm:
    print(readmov)

    moviefolder = os.path.join('CALI_3_'+runfile[3:8]+'_13638_'+read_from_path[-16:-5])
    pth = os.path.join(current_path,moviefolder)
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(pth):
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r, file))
    impath = os.path.join(current_path,moviefolder)
    images_mov = []
    for f in files:
        print(f)
        images_mov.append(imageio.imread(f))
    
    imageio.mimsave(impath+".gif", images_mov)

