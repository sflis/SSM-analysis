#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:44:31 2019

@author: wolkerst
"""


import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
import pickle
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import math
from ssm.core import pchain
from ssm.pmodules import *
from ssm.core.util_pmodules import Aggregate, SimpleInjector
from ssm.pmodules import *
from ssm.pmodules import Reader
from CHECLabPy.plotting.camera import CameraImage
import os
from os import path
import subprocess
import glob


# =============================================================================
# # Load fit parameters
# =============================================================================


def get_fit_parameters(fit_path):
    """ Loads and returns the fit parameter of
        f(x) = ax^2 + bx + d
    """
    property_path = os.path.join(fit_path + "/", "fit_parameters")
    with open(property_path, "rb") as handle:
        props = pickle.load(handle)
    # print(props)
    a = props["all_parameters"][:, 0]
    b = props["all_parameters"][:, 1]
    d = props["all_parameters"][:, 2]

    return (a, b, d)


# =============================================================================
# # Load calibration data - only the data.data and time is necessary
# =============================================================================


def get_your_data(runfile):
    data_proc = pchain.ProcessingChain()
    path = "/data/CHECS-data/astri_onsky/slowsignal/" + str(runfile) + ".hdf5"
    reader = Reader(path)
    data_proc.add(reader)
    frame_cleaner = PFCleaner()
    data_proc.add(frame_cleaner)
    aggr = Aggregate(["raw_resp"])
    data_proc.add(aggr)
    data_proc.run()
    data = aggr.aggr["raw_resp"][0]
    dt = data.time - data.time[0]
    return (data, dt)


def clean_data(data):
    """ Remove nan pixels """
    good_pix = np.ones(data.data.shape, dtype="bool")
    good_pix[np.nan_to_num(data.data) == 0] = False
    return good_pix


def get_smooth_int(
    data,
    dt,
    good_pixel,
    minutes_of_stable_int=None,
    interval_stable=None,
    plot_stable_interval=None,
):
    """
    Input: data (as data.data), dt(timespan), good_pixel
    Optional: minutes_of_stable_int 
              interval_stable 
              plot_stable_interval    
    """
    # Get stable interval from minimum of std, from calibrated data in 4 min measurement intervals
    min_of_run = math.floor(dt[-1] / 60)
    if interval_stable == None:
        if minutes_of_stable_int == None:
            if min_of_run < 1:
                # In case the measurement is very short
                min_of_run = 1
                minutes_of_stable_int = 0.2
            else:
                minutes_of_stable_int = 4
        number_of_intervals = int(min_of_run / minutes_of_stable_int)

        # Splits calibrated data into 4 min intervals
        cal_interval_ = np.array_split(data, number_of_intervals, axis=0)
        cal_interval_std = [
            np.std(cal_interval_[:][i], axis=0) for i in range(number_of_intervals)
        ]
        cal_std = [
            np.nanmean(cal_interval_std[i][good_pixel[0, :] == 1], axis=0)
            for i in range(number_of_intervals)
        ]
        # Dont look at this code ..
        bin_of_min_std = np.where(cal_std == min(cal_std))[0][0]
        min_cal_std = np.array_split(dt, number_of_intervals)[bin_of_min_std]
        int_start = list(dt).index(min_cal_std[0])
        int_stop = list(dt).index(min_cal_std[-1])
    else:
        if minutes_of_stable_int == None:
            if min_of_run < 1:
                # In case the measurement is very short
                min_of_run = 1
                minutes_of_stable_int = 0.2
            else:
                minutes_of_stable_int = 4
        number_of_intervals = int(min_of_run / minutes_of_stable_int)
        min_cal_std = np.array_split(dt, number_of_intervals)[interval_stable]
        int_start = list(dt).index(min_cal_std[0])
        int_stop = list(dt).index(min_cal_std[-1])

    if plot_stable_interval != None:
        # Plot for some pixel to see what is chosen
        plt.figure()
        plt.axvline(dt[int_start], color="k")
        plt.axvline(dt[int_stop], color="k")
        plt.xlabel("Time / s")
        plt.ylabel("Amplitude / (mV)")
        for i in [1, 23, 600, 900, 1200]:
            plt.plot(dt, data[:, i])

    return (int_start, int_stop)


def invf(y, a, b, c):
    return np.nan_to_num(-(b - np.sqrt(b * b - 4 * a * (c - y))) / (2 * a))


def calibrate_data(
    runfile_cal,
    path_,
    save_file=None,
    minutes_of_stable_int=None,
    interval_stable=None,
    plot_stable_interval=None,
):

    """
    Input:      runfile_cal ... data for calibration
                path_ ... place of data
    Optional:   save_file ... no saving by default
                if True ... calibration saved to:
                path_+'/'+ runfile_cal+"calibration_properties"
                minutes_of_stable_int ... default: 4 mins for longer runs and 20s for short runs
                interval_stable ... if you want to control which interval is chosen
                plot_stable_interval ... default None, if any it will plot int for some pixels
 
    Output:     ff_c ... ff-coeficient
                int_begin,int_end ... interval of least RMS
                int_time_averaged ... time avergaed value of interval
                int_space_averaged ... camera averaged
                
    """

    data, dt = get_your_data(runfile_cal)
    good_pixel = clean_data(data.data)  # same shape as data.data

    # Get time interval with stable background:
    intensity = invf(data.data, a, b, d)
    intensity[good_pixel == 0] = -1  # Set bad pix to dummy intensity unequal zero

    int_begin, int_end = get_smooth_int(
        data.data,
        dt,
        good_pixel,
        minutes_of_stable_int,
        interval_stable,
        plot_stable_interval,
    )
    int_time_averaged = np.mean(
        intensity[[int_begin, int_end], :], axis=0
    )  # Mean of each pixel individually over stable int time
    int_space_averaged = np.mean(
        intensity[:, good_pixel[0][:]], axis=1
    )  # mean of all pixels at each time frame
    int_spacetime_averaged = np.mean(int_space_averaged)

    # flat fieding Coefficient:
    ff_c = int_spacetime_averaged / int_time_averaged
    ff_c[good_pixel[0, :] == 0] = 0

    calibrated_data = ff_c * intensity
    # Get the offet of Calibrated data via the mean in a good interval
    # I_offset_of_calibration = I_cal - offset_interval
    offset_calibrated_data = calibrated_data - np.mean(
        calibrated_data[int_begin:int_end], axis=0
    )

    if save_file == True:
        # Save new calibration files
        property_path = os.path.join(
            path_ + "/" + runfile_cal, "calibration_properties"
        )
        calibration_properties = {
            "Interval_offset": (int_begin, int_end),
            "New_good_pixel": good_pixel,
            "Time_averaged_int": int_time_averaged,
            "Space_averaged": int_space_averaged,
            "ff_coefficients_c": ff_c,
            "Offset_calibrated_data": offset_calibrated_data,
        }
        with open(property_path, "wb") as handle:
            pickle.dump(calibration_properties, handle)
    return (ff_c, int_begin, int_end, int_time_averaged, int_space_averaged)


def flat_fielding(
    path_,
    runfile_ff,
    runfile_cal=None,
    calibrate=None,
    save_calibration_file=None,
    save_ff_data=None,
):

    """ 
    Input necessary: path where calibrated data is stored 
                     runfile_ff ... data to flat field
    Optional input:  runfile_cal ... data used for calibration, default "Run13638_ss"
                     calibrate ... any input will lead to recalculation of calibration properties
                     save_calibration_file ... any input save the calibration properties to "path_+calibration_properties"
                     Existing files may be overwritten
                     save_ff_data ... will save the flat_fielding output with file it was calibrated 
                     e.g. "path_+runfile_cal+Calibration_with+runfile_cal"
        
    This function returns:
        - the flat field coefficient ff_c  
        - the calibrated and ff-data F_i
        - the calibrated data I_cal
    
    """

    if runfile_cal == None:
        runfile_cal = "Run13638_ss"
    if calibrate != None:
        ff_c, int_begin, int_end, int_time_averaged, int_space_averaged = calibrate_data(
            runfile_cal, path_, save_calibration_file
        )
    if calibrate == None:
        property_path = os.path.join(
            path_ + "/" + runfile_cal, "calibration_properties"
        )
        with open(property_path, "rb") as handle:
            calibration_data = pickle.load(handle)
        int_begin, int_end = calibration_data["Interval_offset"]
        good_pixel = calibration_data["New_good_pixel"]
        ff_c = calibration_data["ff_coefficients_c"]

    data, dt = get_your_data(runfile_ff)
    good_pixel = clean_data(data.data)  # same shape as data.data
    # Get time interval with stable background:
    I_cal = invf(data.data, a, b, d)
    I_cal[good_pixel == 0] = 0
    # Calibrated and Flat fielded data:
    F_i = ff_c * I_cal

    if save_ff_data != None:
        # Save calibations
        property_path = os.path.join(
            path_ + "/" + runfile_ff, "Calibration_with_" + str(runfile_cal)
        )
        CAL = {
            "Interval_offset": (int_begin, int_end),
            "New_good_pixel": good_pixel,
            "Time_averaged_int": int_time_averaged,
            "Space_averaged": int_space_averaged,
            "Calibrated_data": F_i,
            "ff_coefficients_c": ff_c,
        }
        with open(property_path, "wb") as handle:
            pickle.dump(CAL, handle)

    return (F_i, ff_c, I_cal)


def compare_ffc(path_, runfile_cal_a, runfile_cal_b):
    """ 
    
    Plots the histogram of two ff-coeffs e.g.
    runfile_cal_a = 'Run13637_ss'    
    runfile_cal_b = 'Run13638_ss' 
    
    """
    property_path = os.path.join(path_ + "/" + runfile_cal_a, "calibration_properties")
    with open(property_path, "rb") as handle:
        readfile1 = pickle.load(handle)
    coeff_1 = readfile1["ff_coefficients_c"]
    property_path = os.path.join(path_ + "/" + runfile_cal_b, "calibration_properties")
    with open(property_path, "rb") as handle:
        readfile1 = pickle.load(handle)
    coeff_2 = readfile1["ff_coefficients_c"]

    compare_c = np.nan_to_num((coeff_1 - coeff_2) / coeff_2)
    plt.figure(figsize=(8, 6))
    plt.hist(compare_c, bins=100, range=(-0.1, 0.1))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(
        [
            r"$\frac{c_{"
            + str(runfile_cal_a)
            + "} - c_{"
            + str(runfile_cal_b)
            + "}}{c_{"
            + str(runfile_cal_b)
            + "}}$ "
        ],
        fontsize=19,
        loc=1,
    )
    plt.axvline(0, c="k")


def plot_calibration_properties(
    path_,
    runfile_cal,
    save=None,
    camera_min=None,
    camera_max=None,
    cal_min=None,
    cal_max=None,
):

    # import data:
    data, dt = get_your_data(runfile_cal)

    # Import calibration properties
    property_path = os.path.join(path_ + "/" + runfile_cal, "calibration_properties")
    with open(property_path, "rb") as handle:
        readout = pickle.load(handle)
    int_begin, int_end = readout["Interval_offset"]
    int_time_averaged = readout["Time_averaged_int"]
    int_space_averaged = readout["Space_averaged"]
    calibrated_data = readout["Calibrated_data"]
    ff_c = readout["ff_coefficients_c"]
    offset_calibrated_data = readout["Offset_calibrated_data"]

    if camera_min == None:
        zmin_intspace = min(int_space_averaged) - 0.05 * min(int_space_averaged)
    else:
        zmin_intspace = camera_min
    if camera_max == None:
        zmax_intspace = max(int_space_averaged) + 0.05 * max(int_space_averaged)
    else:
        zmax_intspace = camera_max

    if cal_min == None:
        zmin_calbdat = min(int_space_averaged) - 0.01 * min(int_space_averaged)
    else:
        zmin_calbdat = cal_min
    if cal_max == None:
        zmax_calbdat = max(int_space_averaged) + 0.01 * max(int_space_averaged)
    else:
        zmax_calbdat = cal_max

    # Visualize for some pixels:
    plt.figure()
    plt.axvline(dt[int_begin], color="k")
    plt.axvline(dt[int_end], color="k")
    plt.xlabel("Time / s")
    plt.ylabel("Amplitude / (mV)")
    plt.title("Interval of offset")
    for i in [1, 23, 600, 900, 1200]:
        plt.plot(dt, calibrated_data[:, i])
    if save != None:
        plt.savefig(
            os.path.join(
                current_path + "/" + runfile_cal, runfile_cal + "_std_interval"
            )
        )

    plt.figure()
    plt.plot(data.time - data.time[0], int_space_averaged)
    plt.xlabel("Time since run start (s)")
    plt.ylabel("Average amplitude (mV)")
    if save != None:
        plt.savefig(
            os.path.join(
                current_path + "/" + runfile_cal,
                runfile_cal + "_space_averaged_over_time",
            )
        )

    camera = CameraImage(data.xpix, data.ypix, data.size)
    camera.image = int_time_averaged
    camera.set_limits_minmax(zmin=zmin_intspace, zmax=zmax_intspace)
    camera.add_colorbar("Amplitdue (mV)")
    camera.ax.set_title("Time averaged data")
    if save != None:
        plt.savefig(
            os.path.join(
                current_path + "/" + runfile_cal, runfile_cal + "_camera_time_averaged"
            )
        )

    camera = CameraImage(data.xpix, data.ypix, data.size)
    camera.image = calibrated_data[0, :]
    camera.add_colorbar("Amplitdue (mV)")
    zmin_calbdat = min(int_space_averaged) - 0.01 * min(int_space_averaged)
    # zmin_calbdat = 380
    zmax_calbdat = max(int_space_averaged) + 0.01 * max(int_space_averaged)
    camera.set_limits_minmax(zmin=zmin_calbdat, zmax=zmax_calbdat)
    camera.ax.set_title("Calibrated Data")
    if save != None:
        plt.savefig(
            os.path.join(
                current_path + "/" + runfile_cal, runfile_cal + "_calibrated_data"
            )
        )

    camera = CameraImage(data.xpix, data.ypix, data.size)
    camera.image = ff_c
    camera.add_colorbar("Amplitdue (mV)")
    camera.ax.set_title("Flat field coefficents $ff_{c}$")
    if save != None:
        plt.savefig(
            os.path.join(
                current_path + "/" + runfile_cal, runfile_cal + "_flat_field_coeffs_c"
            )
        )

    camera = CameraImage(data.xpix, data.ypix, data.size)
    camera.image = offset_calibrated_data[0, :]
    camera.add_colorbar("Amplitdue (mV)")
    camera.ax.set_title("Offset of calibrated data")
    zmin_offset = None
    zmax_offset = None
    # np.where(offset_calibrated_data > 50)
    # zmin_offset = 0
    # zmax_offset = 60
    camera.set_limits_minmax(zmin=zmin_offset, zmax=zmax_offset)
    if save != None:
        plt.savefig(
            os.path.join(
                current_path + "/" + runfile_cal,
                runfile_cal + "_offset_calibrated_data",
            )
        )


def get_time(frame):
    #
    m, s = divmod(frame, 60)
    h, m = divmod(m, 60)

    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


############## Try out functions ###################################################

# Suspicious patients
# bad_pixel = np.array([25, 58, 101, 226, 256, 259, 304, 448, 449,570, 653, 670, 776, 1049, 1094,
#                      1158, 1177, 1352, 1367, 1381,1427, 1434,1439,1503, 1562, 1680, 1765,
#                      1812, 1813, 1814, 1815, 1829, 1852, 1869, 1945, 1957, 2009, 2043])


# current_path = '/home/wolkerst/projects/cta/SSM-analysis' ## Change to where it will finally be
current_path = os.getcwd()
a, b, d = get_fit_parameters(current_path)

# Define which run to calibrate with which
# Default calibration run
runfile_cal = "Run13638_ss"
# Data for flat fielding
runfile_ff = "Run13312"

flat_fielding(os.getcwd(), "Run13312")


def get_time(frame):

    m, s = divmod(frame, 60)
    h, m = divmod(m, 60)

    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
