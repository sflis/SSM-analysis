import numpy as np


def find_unstable_pixs(res,time,rms_thresh = 40.0):
    diff_rms = np.std(np.abs(np.diff(res,axis=0)/np.diff(time)[:,np.newaxis]),axis=0)
    return np.where(diff_rms>rms_thresh)[0]


def smooth_slowsignal(a, n=10):
    """ Simple smoothing algorithm that uses a moving average between readout frames
        It assumes that the time between each readout is equidistant.
    """
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n
