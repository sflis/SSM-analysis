from ssm.fit.fit import FitModel, FitParameter

from astropy.time import Time
import numpy as np
from copy import deepcopy
class DataFeeder(FitModel):
    def __init__(self, data, times):
        super().__init__("FitStarter")
        self.times = Time(times, format="unix")
        self.data = deepcopy(data)
        ci = []
        for k,v in self.data.items():

            ti,res = zip(*v)
            self.data[k] = (np.array(ti,dtype=np.uint64),np.array(res))

            if v[0][0] == 0:
                ci.append(k)
        self.cluster_i = ci#[k if v[0][0]==0 ]
        self.out_data = "data"
        self.configured = False
    def configure(self, config):
        if not self.configured:
            config["time_steps"] = self.times
            config["n_frames"] = 1
            config["cluster_i0"] = self.cluster_i
    def run(self, frame):
        frame[self.out_data] = self.data
        return frame



from collections import defaultdict
from ssm.star_cat import hipparcos
class IlluminationModel(FitModel):
    def __init__(self, pix_model):
        super().__init__("IlluminationModel")
        self.pix_model = pix_model
        self.in_sources = "star_sources"
        self.out_raw_response = "raw_response"
        self.in_data = 'data'
        self.par_effmirrorarea = FitParameter("effmirrorarea", 6.5, [0.0, 8.0])

    def configure(self, config):
        self.lc = hipparcos.LightConverter()
        self.pix_pos = config["pix_pos"]

    def run(self, frame):
        star_srcs = frame[self.in_sources]
        data = frame[self.in_data]
        res = {}
        rates = self.lc.mag2photonrate(star_srcs.vmag, area=self.par_effmirrorarea.val)* 1e-6
        for pix,v in data.items():
            ti = v[0]
            pix_resp = np.zeros(ti.shape)#,ti.copy())
            ppos = self.pix_pos[pix]
            for star,rate in zip(star_srcs.pos,rates):
                l = np.linalg.norm(star[ti] - ppos, axis=1)
                i = np.where(l < self.pix_model.model_size)[0]
                x = star[:,0][ti][i] - ppos[0]
                y = star[:,1][ti][i] - ppos[1]
                pix_resp[i] += self.pix_model(x, y, grid=False) * rate
            res[pix] = [ti.copy(),pix_resp]
        frame[self.out_raw_response] = res
        return frame

class FlatNSB(FitModel):
    def __init__(self):
        super().__init__("FlatNSB")
        self.in_raw_response = "raw_response"
        self.out_raw_response = "raw_response"
        self.par_nsbrate = FitParameter("nsbrate", 40.0, [0.0, 100.0])

    def configure(self, config):
        pass

    def run(self, frame):
        res = frame[self.in_raw_response]
        for pix,v in res.items():
            res[pix][1]
            res[pix][1] += self.par_nsbrate.val
        frame[self.in_raw_response] = res

        return frame


class Response(FitModel):
    def __init__(self, calib):
        super().__init__("Response")
        self.calib = calib
        self.in_raw_response = "raw_response"
        self.out_response = "response"

    def configure(self, config):
        pass

    def run(self, frame):
        resp = {}
        inresp = frame[self.in_raw_response]
        for pix,v in inresp.items():
            respv = [v[0].copy(),None]
            respv[1] = self.calib.rate2mV(v[1])
            resp[pix] = respv
        frame[self.out_response] = resp
        return frame


