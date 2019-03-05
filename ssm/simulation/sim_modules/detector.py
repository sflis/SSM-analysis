from ssm.core.pchain import ProcessingModule
import numpy as np
from ssdaq.core.ss_data_classes import ss_mappings
from ssm.star_cat import hipparcos

class DetectorResponse(ProcessingModule):
    def __init__(self,pix_model,calib):
        super().__init__('DetectorResponse')
        self.pix_model = pix_model
        self.calib = calib
    def configure(self,config):
        # Computing inverse mapping for full camera
        self.mapping = np.empty((32,64),dtype=np.uint64)
        invm=np.empty(64)
        for i,k in enumerate(ss_mappings.ssl2asic_ch):
            invm[k] = i
        for i in range(32):
            self.mapping[i,:] = invm+i*64
        self.mapping = self.mapping.flatten()
        self.lc = hipparcos.LightConverter()
        self.pix_pos = config['pix_pos']
    def run(self,frame):
        res = np.zeros(2048)
        star_srcs = frame['star_sources']
        for spos, rate in zip(star_srcs.pos,
                            self.lc.mag2photonrate(star_srcs.vmag,
                                                    area=6.5)*1e-6):
            self._raw_response(spos,res,rate)
        frame['raw_response'] = res[self.mapping]
        frame['response'] = self.calib.rate2mV(frame['raw_response'])
        return frame

    def _raw_response(self,source,res,rate):
        # res = np.zeros(self.pix_posy.shape)# if res is None else res
        l = np.linalg.norm(source-self.pix_pos,axis=1)
        i = np.where(l < self.pix_model.model_size)[0]
        x = source[0]-self.pix_pos[i][:,0]
        y = source[1]-self.pix_pos[i][:,1]
        res[i] += self.pix_model(x,y,grid=False)*rate