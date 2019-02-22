import numpy as np
from scipy import interpolate
import pynverse
#A temporary place to put the model based on data taken by Steve Leach at MPIK

f = 0.736#conversion factor from LED rate setting to real rate
#data
nsb_MHz = np.array([0, 6, 10, 20, 29, 38, 43, 52, 58, 90*f, 100*f])
mean_mV = np.array([0, 0, 0, 1.5, 10.5, 30.5, 56, 81, 104, 126, 146])


pol = np.polyfit(nsb_MHz[nsb_MHz>35],mean_mV[nsb_MHz>35],deg=1)

m = (nsb_MHz>6) & (nsb_MHz<30)

rate_mV = interpolate.CubicSpline(list(nsb_MHz[m])+[40],list(mean_mV[m])+[np.polyval(pol,40)],bc_type=((1, 0.0), (1, pol[0])),extrapolate=False)
rate = np.linspace(0,100,100)


def rate2mV(x):
    x = np.asarray(x)
    mV = np.empty(x.shape)

    m = x<=10
    if(np.any(m)):
        mV[m] = 0
    m =(x>10)&(x<40)
    if(np.any(m)):
        mV[m] = rate_mV(x[m])
    m = (x>40)
    if(np.any(m)):
        mV[m] = np.polyval(pol,x[m])
    return mV

mV2rate = pynverse.inversefunc(rate2mV,domain=[0,4000])
