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

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import copy
from CHECLabPy.plotting.camera import CameraImage
def make_ssmovie(data,
                red,
                highlightpix=None,
                minmax=(-10,
                    100),
                path='movie',
                dpi=800,
                scale=0.2,
                title="",
                fps=25,
                filename="out",
                zlabel="Amplitude (mV)"):
    """Summary

    Args:
        data (TYPE): A SlowsignalData object
        red (TYPE): A range object or a list of indices
        highlightpix (None, optional): A list of pixel highlight index arrays
        minmax (tuple, optional): min max for the z-scale
        path (str, optional): The path from the current working directory where to store images for the movie
        dpi (int, optional): Description
        scale (float, optional): Description
        title (str, optional): The title to be shown
        fps (int, optional): Number of frames per second in the movie
        filename (str, optional): output name of the movie
        zlabel (str, optional): The colorbar label
    """
    import os
    import subprocess

    import glob
    impath = os.path.join(os.getcwd(),path)
    if not os.path.isdir(impath):
        os.mkdir(impath)

    files = glob.glob(os.path.join(impath,"*"))
    for f in files:
        os.remove(f)

    dpi -= int(scale*19.20*dpi)%2
    dpii =400
    scale = 0.70
    fig,ax = plt.subplots(figsize=(1920/dpii*scale,1080/dpii*scale),dpi=dpii*scale)
    camera = CameraImage(data.xpix, data.ypix, data.size,ax=ax)


    camera.add_colorbar(zlabel)
    camera.set_limits_minmax(*minmax)
    im = copy.deepcopy(data.data[0])
    camera.image = im
    highl = None
    for i in tqdm(red,total=len(red)):
        im = copy.deepcopy(data.data[i])
        im[np.isnan(im)] = np.nanmean(im)

        camera.ax.set_title(title)
        if highl is None:
            highl = camera.highlight_pixels(highlightpix[i])
        else:
            lw_array = np.zeros(camera.image.shape[0])
            lw_array[highlightpix[i]] = 0.5
            highl.set_linewidth(lw_array)

        camera.image = im
        plt.savefig(os.path.join(path,"SlowSignalImage%.10d.png"%i),dpi=dpi)
    subprocess.check_call(["ffmpeg",
                     "-pattern_type","glob",
                     "-i",
                     "{}".format(os.path.join(impath,"SlowSignalImage*.png")),
                     "-c:v", "libx264","-vf"," scale=iw:-2", "-vf", "fps={}".format(fps),
                           "-pix_fmt",
                           "yuv420p",'-y', "{}.mp4".format(filename)],cwd = os.getcwd())