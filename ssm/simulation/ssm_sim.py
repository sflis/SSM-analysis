from collections import namedtuple
import datetime
import numpy as np
from tqdm.auto import tqdm

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

from ctapipe.coordinates import CameraFrame, HorizonFrame
from ctapipe.instrument import CameraGeometry

import target_calib
from ssm.models import calibration
from ssm.star_cat import hipparcos
from ssdaq.core.ss_data_classes import ss_mappings
from ssdaq import SSReadout
import ssm
from ssm.utils.model_tools import Line
from ssm.io.sim_io import SimDataWriter,SourceDescr
_c = target_calib.CameraConfiguration("1.1.0")
_mapping = _c.GetMapping()
xv = np.array(_mapping.GetXPixVector())*1e3
yv = np.array(_mapping.GetYPixVector())*1e3

class ModularSim:
    def __init__(self):
        self.chain = []
        self.config = {}
        self.frame_n = 0
    def add(self, module):
        self.chain.append(module)

    def __str__(self):
        s = ''
        for m in self.chain:
            s += m.__str__()+'\n'
        return s
    def configure(self):
        self.config = {}
        self.frame_n = 0
        for module in self.chain:
            module.configure(self.config)

    def mod_list(self):
        for mod in self.chain:
            print(mod.name)

    def run(self,max_frames = None):
        n_frames = self.config['n_frames'] if max_frames is None else max_frames
        for i in tqdm(range(n_frames)):
            self.frame_n += 1
            frame = {'frame_n':self.frame_n}
            for module in self.chain:
                frame =module.run(frame)

class SimModule:
    def __init__(self,name):
        self._name = name

    def configure(self,config):
        raise NotImplementedError

    def run(self,frame):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

class Time(SimModule):
    def __init__(self,start_time,end_time,time_step=datetime.timedelta(seconds=.1)):
        super().__init__('Time')
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = time_step
        self.n_calls = 0
    def configure(self,config):
        config['start_time'] = self.start_time
        config['end_time'] = self.end_time
        config['time_step'] = self.time_step
        config['run_duration'] = self.end_time - self.start_time
        config['n_frames'] = int(config['run_duration'].to_datetime().total_seconds()/self.time_step.total_seconds())
        self.n_calls = 0
    def run(self,frame):
        frame['time'] = self.start_time + self.time_step*self.n_calls
        self.n_calls +=1
        return frame

class Print(SimModule):
    def __init__(self,name,print_keys=None,sel=1):
        super().__init__(name)
        self.print_keys = print_keys
        self.sel = sel
        self.n_frames = 0
    def configure(self,config):
        self.n_frames = 0
        pass

    def run(self,frame):
        if(self.n_frames%self.sel == 0):
            print('+++++ %s +++++'%self.name)
            for k,v in frame.items():
                print("{}: {}".format(k,v))
            # print(frame)
        self.n_frames += 1
        return frame

class Telescope(SimModule):
    def __init__(self,
                location= EarthLocation.from_geodetic(lon = 14.974609,
                                                    lat = 37.693267,
                                                    height=1730),
                cam_config = _c,
                focal_length = 2.15,
                target = None,
                fov = 5
                ):
        super().__init__("Telescope")
        self.location = location
        self.cam_config = cam_config
        self.focal_length = focal_length
        self._mapping = self.cam_config.GetMapping()
        self.pix_posx = np.array(self._mapping.GetXPixVector())*1e3#mm
        self.pix_posy = np.array(self._mapping.GetYPixVector())*1e3#mm
        self.pix_pos = np.array(list(zip(self.pix_posx,self.pix_posy)))
        self.target = target
        self.fov = fov

    def configure(self,config):
        config['pix_pos'] = self.pix_pos
        config['fov'] = self.fov

        td = config['time_step']
        n_frames = config['n_frames']
        #precomputing transformations
        #Will need to do this smarter when number of frames reaches 10k-100k
        self.obstimes = np.array([config['start_time'] +td*i for i in range(n_frames)])
        self.precomp_hf = HorizonFrame(location=self.location, obstime=self.obstimes)
        self.precomp_point = self.target.transform_to(self.precomp_hf)
        self.precomp_camf = CameraFrame(
            telescope_pointing=self.precomp_point,
            focal_length=u.Quantity(self.focal_length, u.m),#28 for LST
            obstime=self.obstimes,
            location=self.location,
        )
        config['start_pointing'] = self.precomp_point[0]
    def run(self,frame):
        obstime = frame['time']
        horizon_frame = self.precomp_hf[frame['frame_n']-1]
        pointing = self.precomp_point[frame['frame_n']-1]
        current_cam_frame = self.precomp_camf[frame['frame_n']-1]
        frame['horizon_frame'] = horizon_frame
        frame['pointing'] = pointing
        frame['current_cam_frame'] = current_cam_frame
        return frame


class Stars(SimModule):
    def __init__(self,stars,vmag_lim = (0,9)):
        super().__init__('Stars')
        self.stars = stars
        self.stars_in_fov = None
        self.fov_star_mask = None
        self.star_coords = SkyCoord(ra=stars.ra_deg.values,dec=stars.dec_deg.values,unit='deg')
        self.vmag_lim = vmag_lim
        self._StarSources = namedtuple("StarSources",'pos vmag')

    def configure(self,config):
        self.fov = config['fov']
        self._get_stars_in_fov(config['start_pointing'])

    def _get_stars_in_fov(self,pointing):
        target = pointing.transform_to('icrs')
        # stars_in_fov
        s = target.separation(self.star_coords)
        self.fov_mask = (s.deg<self.fov) & (self.stars.vmag<self.vmag_lim[1])

    def run(self,frame):
        s_cam = self.star_coords[self.fov_mask].transform_to(frame['current_cam_frame'])

        frame['star_sources'] = self._StarSources(np.array([s_cam.x.to_value(u.mm),
                                                            s_cam.y.to_value(u.mm)]).T,
                                                    self.stars.vmag[self.fov_mask])
        return frame

class DetectorResponse(SimModule):
    def __init__(self,pix_model,calib):
        super().__init__('DetectorResponse')
        self.pix_model = pix_model
        self.calib = calib
    def configure(self,config):
        # Determining mapping for full camera
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
        frame['raw_response'] = res
        frame['raw_response'] = self.calib.rate2mV(res[self.mapping])
        return frame

    def _raw_response(self,source,res,rate):
        # res = np.zeros(self.pix_posy.shape)# if res is None else res
        l = np.linalg.norm(source-self.pix_pos,axis=1)
        i = np.where(l < self.pix_model.model_size)[0]
        x = source[0]-self.pix_pos[i][:,0]
        y = source[1]-self.pix_pos[i][:,1]
        res[i] += self.pix_model(x,y,grid=False)*rate
