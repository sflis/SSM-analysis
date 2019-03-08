import numpy as np
from scipy import interpolate
import pynverse

import sys, inspect

class RateCalibration:
    def __init__(self,cal_model,**kwargs):
        self.name = cal_model
        if(cal_model in self.cal_list()):
            self.cal = getattr(sys.modules[__name__], cal_model)(**kwargs)
        else:
            raise ValueError('No calibration named {} found'.format(cal_model))

    @classmethod
    def cal_list(cls):
        cals = []
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj) and name is not cls.__name__:
                cals.append(name)
        return cals

    def rate2mV(self,rate):
        return self.cal.rate2mV(rate)

    def mV2rate(self,mV):
        return self.cal.mV2rate(mV)


class SteveCal:
    def __init__(self):
        #A temporary place to put the model based on data taken by Steve Leach at MPIK

        self.f = 0.736#conversion factor from LED rate setting to real rate
        #data
        self.nsb_MHz = np.array([0, 6, 10, 20, 29, 38, 43, 52, 58, 90*self.f, 100*self.f])
        self.mean_mV = np.array([0, 0, 0, 1.5, 10.5, 30.5, 56, 81, 104, 126, 146])


        self.pol = np.polyfit(self.nsb_MHz[self.nsb_MHz>35],self.mean_mV[self.nsb_MHz>35],deg=1)

        m = (self.nsb_MHz>6) & (self.nsb_MHz<30)

        self.rate_mV = interpolate.CubicSpline(list(self.nsb_MHz[m])+[40],
                                        list(self.mean_mV[m])+[np.polyval(self.pol,40)],
                                        bc_type=((1, 0.0), (1, self.pol[0])),
                                        extrapolate=False)
        self.mV2rate= pynverse.inversefunc(self.rate2mV,domain=[0,4000])

    def rate2mV(self,x):
        x = np.asarray(x)
        mV = np.empty(x.shape)

        m = x<=10
        if(np.any(m)):
            mV[m] = 0
        m =(x>10)&(x<=40)
        if(np.any(m)):
            mV[m] = self.rate_mV(x[m])
        m = (x>40)
        if(np.any(m)):
            mV[m] = np.polyval(self.pol,x[m])
        return mV

class SteveCalCutoff():
    def __init__(self, mV_cutoff=3000.):
        self.cal = SteveCal()#super().__init__()
        self.mV_cutoff = mV_cutoff
        self.rate_cutoff = self.cal.mV2rate(self.mV_cutoff)
        self.mV2rate= pynverse.inversefunc(self.rate2mV,domain=[0,self.mV_cutoff+10])
    def rate2mV(self,x):
        x = np.asarray(x)
        mV = np.empty(x.shape)
        m = x > self.rate_cutoff

        if np.any(m) :
            mV[m] = self.mV_cutoff
        if np.any(m):
            mV[~m] = self.cal.rate2mV(x[~m])
        return mV


class SteveCalSmoothCutoff():
    def __init__(self, mV_cutoff_start=1000.):
        self.cal = SteveCal()
        self.start_rate = self.cal.mV2rate(mV_cutoff_start)
        self.end_rate = 3000

        self.cutoff_val = 3000
        self.rate_mV = interpolate.CubicSpline([self.start_rate,2000,2850,self.end_rate],
                                        [mV_cutoff_start,2970,2999,self.cutoff_val],
                                        bc_type=((1, self.cal.pol[0]), (1, 0.)),
                                        extrapolate=False)

        self.mV2rate= pynverse.inversefunc(self.rate2mV,domain=[0,self.cutoff_val+10])

    def rate2mV(self,x):
        x = np.asarray(x)
        mV = np.empty(x.shape)
        m = (x > self.start_rate) & (x<=self.end_rate)
        if np.any(m) :
            mV[m] = self.rate_mV(x[m])
        m = x>self.end_rate
        if np.any(m):
            mV[m] = self.cutoff_val

        m = x <= self.start_rate
        if np.any(m):
            mV[m] = self.cal.rate2mV(x[m])

        return mV