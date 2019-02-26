import numpy as np
# from tqdm import tqdm
from tqdm.auto import tqdm
import target_calib

from ssdaq.core.ss_data_classes import ss_mappings
from ssdaq import SSReadout
import ssm
from ssm.utils.model_tools import Line
from ssm.io.sim_io import SimDataWriter,SourceDescr
_c = target_calib.CameraConfiguration("1.1.0")
_mapping = _c.GetMapping()
xv = np.array(_mapping.GetXPixVector())*1e3
yv = np.array(_mapping.GetYPixVector())*1e3

class SingleSource:
    def __init__(self,rate,name,vmag=None,radec=None):
        self.rate = rate
        self.p = None
        self.vmag = vmag
        self.radec = radec
        self.name = name
    def advance(self):
        pass

    def __str__(self):
        s = "<{}: name={}, p=({},{}), rate={}, vmag={}, radec={}>".format(self.name,
                                                                            self.p,
                                                                            self.vmag,
                                                                            self.radec)

        return s



class LineSource(SingleSource):
    def __init__(self,name, start, stop,v,dt, rate):
        super().__init__(rate,name)
        self.start = np.array(start,dtype=np.float64)
        self.stop = np.array(stop,dtype=np.float64)
        self.length = np.linalg.norm(self.start-self.stop)
        self.line = Line(start,stop)
        self.p = self.start
        self.v = v
    def advance(self,N,dt):

        for i in range(N):
            self.p = self.line(np.array([i*self.v*dt]))
            yield self

class Sources:
    def __init__(self,source_list,dt,N):
        self.source_list = source_list
        self.dt = dt
        self.N = N
    def advance(self):
        sources = []
        for s in self.source_list:
            sources.append(s.advance(self.N,self.dt))
        for i in range(self.N):
            ret = []
            for s in sources:
                ret.append(s.__next__())
            yield ret


class FocalPlaneModel:
    def __init__(self,pix_resp,pix_posx=xv,pix_posy=yv):
        self.pix_resp = pix_resp
        self.pix_posx = pix_posx
        self.pix_posy = pix_posy
        self.pix_pos = np.array(list(zip(pix_posx,pix_posy)))


    def _raw_response(self,source):
        res = np.zeros(self.pix_pos.shape)# if res is None else res
        # for s in sources:
        for j, pp in enumerate(self.pix_pos):
            if np.linalg.norm(s-pp) > self.pix_resp.model_size:#self.pix_resp.m_size : #FIXME: corners not considered yet
                continue
            res[j] = self.pix_resp(s[0]-pp[0],s[1]-pp[1])
        res[np.repeat(ss_mappings.ssl2asic_ch,32)] = res[:]
        return res

    def response2(self,sources):
        res = []
        for adv_srcs  in sources.advance():
            tres = np.zeros(self.pix_pos.shape)
            res.append(tres)
            for s in adv_srcs:
                tres += rate2mV(self._raw_response(s.p)*s.rate)

        return res

    def response(self,paths,rate):
        from ssm.models.rate2amp_conv import rate2mV
        l =[]
        for p in paths:
            l.append(len(p))
        res = np.zeros((max(l),len(self.pix_pos)))
        paths = np.asarray(paths)
        for i, path in enumerate(tqdm(paths,total=len(paths))):
            res += rate2mV(self.raw_response(path,np.zeros((max(l),len(self.pix_pos))))*rate[i])

    def raw_response(self,path,res = None):
        path = np.asarray(path)
        res = np.zeros((len(path),len(self.pix_pos))) if res is None else res
        for i, p in enumerate(tqdm(path,total=len(path))):
            for j, pp in enumerate(self.pix_pos):
                if np.linalg.norm(p-pp) > self.pix_resp.model_size:#self.pix_resp.m_size : #FIXME: corners not considered yet
                    continue

                res[i,j] = self.pix_resp(p[0]-pp[0],p[1]-pp[1])
            res[i,np.repeat(ss_mappings.ssl2asic_ch,32)] = res[i,:]
        return res

class FocalPlaneModelSim(FocalPlaneModel):
    def __init__(self,pix_resp,pix_posx=xv,pix_posy=yv):
        super().__init__(pix_resp,pix_posx,pix_posy)




    def _open_file(self):
        sim_attrs = {
             'px_mod':{'val':self.pix_resp.response,'xi':self.pix_resp.xi,'yi':self.pix_resp.yi}
            }
        if(self.pix_resp.psf is not None):
            delta = 0.025
            x = y = np.arange(-self.pix_resp.model_size, self.pix_resp.model_size, delta)
            X, Y = np.meshgrid(x, y)
            sim_attrs['psf'] = {'x':x,'y':y,'val':self.pix_resp.psf(X,Y)},

        vMag = 337
        writer = ssm.io.sim_io.SimDataWriter(self.filename,
                                             sim_sources = (SourceDescr('testsource',0,0,vMag),)                                                   ,
                                             sim_attrs = sim_attrs)






class CameraModel:
    def __init__(self,pix_resp, pix_posx=xv,pix_posy=yv):#,pix_size,prate2mV):
        self.pix_resp = pix_resp
        self.pix_posx = pix_posx
        self.pix_posy = pix_posy
        self.pix_pos = np.array(list(zip(pix_posx,pix_posy)))


    def _raw_response(self,source):
        res = np.zeros(self.pix_pos.shape)# if res is None else res
        # for s in sources:
        for j, pp in enumerate(self.pix_pos):
            if np.linalg.norm(s-pp) > self.pix_resp.model_size:#self.pix_resp.m_size : #FIXME: corners not considered yet
                continue
            res[j] = self.pix_resp(s[0]-pp[0],s[1]-pp[1])
        res[np.repeat(ss_mappings.ssl2asic_ch,32)] = res[:]
        return res

    def response(self,sources):
        res = []
        for adv_srcs  in sources.advance():
            tres = np.zeros(self.pix_pos.shape)
            res.append(tres)
            for s in adv_srcs:
                tres += rate2mV(self._raw_response(s.p)*s.rate)

        return res

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from collections import namedtuple
from ctapipe.coordinates import CameraFrame, HorizonFrame
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ssm.models.rate2amp_conv import rate2mV
import datetime
from ssm.star_cat import hipparcos
class StarSource(SingleSource):
    def __init__(self,rate,name, cam_frame_callback,full_cat_info):
        super().__init__(rate,name)
        self.cam_frame_callback = cam_frame_callback
        self.full_cat_info = dict(full_cat_info)
        self.skycoord = SkyCoord(ra=float(full_cat_info.ra_deg),
                                dec=float(full_cat_info.dec_deg),
                                unit='deg')

    def advance(self,N,dt):

        for i in range(N):
            star_cam = self.skycoord.transform_to(self.cam_frame_callback())
            self.p = np.array([star_cam.x.to_value(u.mm),star_cam.y.to_value(u.mm)])
            yield self

class SSMonitorSimulation:
    def __init__(self, pix_mod,
                        cam_config=_c,
                        stars= None,
                        location = EarthLocation.from_geodetic(lon = 14.974609,
                                                                lat = 37.693267,
                                                                height=1730)):
        self.location = location
        self._cam_config =cam_config
        self.pix_mod =pix_mod
        self.stars =stars
        self.star_coord = SkyCoord(ra=stars.ra_deg.values,dec=stars.dec_deg.values,unit='deg')
        self._mapping = self._cam_config.GetMapping()
        self.pix_posx = np.array(self._mapping.GetXPixVector())*1e3#mm
        self.pix_posy = np.array(self._mapping.GetYPixVector())*1e3#mm
        self.pix_pos = np.array(list(zip(self.pix_posx,self.pix_posy)))
        self.user_sources = []
        self.sim_settings = None
        self.current_stars = None
        self.pointing = None
        self.start_time = None
        self.end_time = None
        self.filename = None
        self.time_step = None
        # Determining mapping for full camera
        self.mapping = np.empty((32,64),dtype=np.uint64)
        invm=np.empty(64)
        for i,k in enumerate(ss_mappings.ssl2asic_ch):
            invm[k] = i
        for i in range(32):
            self.mapping[i,:] = invm+i*64
        self.mapping = self.mapping.flatten()
    def setup_sim(self,target,start_time,end_time,time_step=datetime.timedelta(seconds=.1),max_vmag=9.0,filename = None):
        SimRunSettings = namedtuple('SimRunSettings','current_stars target start_time end_time filename time_step run_duration')
        s = target.separation(self.star_coord)
        m = (s.deg<5.0) & (self.stars.vmag<max_vmag)
        current_stars = []+self.user_sources
        lc = hipparcos.LightConverter()
        if(self.stars is not None):
            sel = self.stars[m]

            for i in sel.index:
                star = self.stars.loc[i]
                # print(star.name)
                current_stars.append(StarSource(lc.mag2photonrate(star.vmag,area=6.5),star.name,self._get_cam_frame,star))
        run_duration = end_time - start_time
        self.sim_settings = SimRunSettings(current_stars,
                                            target,
                                            start_time,
                                            end_time,
                                            filename,
                                            time_step,
                                            run_duration)

        print("Simulation setup:")
        print("  Start time: {}".format(start_time))
        print("  End time: {}".format(end_time))
        print("  Duration: {}h".format(run_duration.to_datetime()))
        # print("  Slow Signal sampling rate: {} ".format(run_duration.to_datetime()))
        print("  Number of frames to simulate: {}".format(int(self.sim_settings.run_duration.to_datetime().total_seconds()/self.sim_settings.time_step.total_seconds())))
        print("  Number of sources: {}".format(len(current_stars)))
        print("  File name: {}".format(filename))

    def _get_cam_frame(self):
        return self.current_cam_frame

    def _advance_cam_frame(self,obstime):
        horizon_frame = HorizonFrame(location=self.location, obstime=obstime)
        pointing = self.sim_settings.target.transform_to(horizon_frame)
        self.current_cam_frame = CameraFrame(
            telescope_pointing=pointing,
            focal_length=u.Quantity(2.15, u.m),#28 for LST
            obstime=obstime,
            location=self.location,
        )

    def _open_file(self):
        srcs = []
        for s in self.sim_settings.current_stars:
            srcs.append(SourceDescr(s.name,
                                    float(s.full_cat_info['ra_deg']),
                                    float(s.full_cat_info['dec_deg']),
                                    float(s.full_cat_info['vmag'])))

        self.writer = ssm.io.sim_io.SimDataWriter(self.sim_settings.filename,
                                        sim_sources = srcs                                                   ,
                                        sim_attrs = None,
                                        buffer = 10)


    def run_sim(self):
        n_steps = int(self.sim_settings.run_duration.to_datetime().total_seconds()/self.sim_settings.time_step.total_seconds())
        srcs = Sources(self.sim_settings.current_stars,
                        self.sim_settings.time_step.total_seconds(),
                        n_steps)

        self.cur_obstime = self.sim_settings.start_time
        self._advance_cam_frame(self.cur_obstime)
        if(self.sim_settings.filename is not None):
            self._open_file()
        try:
            res = []
            for nro,adv_srcs  in tqdm(enumerate(srcs.advance()),total=n_steps):
                tres = np.zeros(self.pix_posy.shape)
                paths = []
                for s in adv_srcs:
                    self._raw_response(s.p,tres,s.rate)#tres += self._raw_response(s.p)*s.rate
                    paths.append(s.p)

                # tres = tres.reshape((32,64))
                # print(tres[19])
                # print(np.argmax(tres[19]))
                # print(ss_mappings.ssl2asic_ch[np.argmax(tres[19])])
                # print(np.where(ss_mappings.ssl2asic_ch==np.argmax(tres[19])))
                # print(np.max(tres))

                # tres = tres[self.mapping]#[self.mapping]
                # tres[:,ss_mappings.ssl2asic_ch] = tres

                    # print(tres[19])
                    # tres = tres.flatten()
                    # print(np.max(tres))
                res.append(rate2mV(tres[self.mapping]))
                if(self.sim_settings.filename is not None):
                    curr_duration = self.cur_obstime - self.sim_settings.start_time
                    timestamp = curr_duration.to_datetime().total_seconds()*1e9
                    self.writer.write_readout(SSReadout(nro+1, timestamp,tres.reshape((32,64))),paths)

                self.cur_obstime = self.cur_obstime + self.sim_settings.time_step
                self._advance_cam_frame(self.cur_obstime)
                # break
        except Exception as e:
            print(e)
        finally:
            if(self.sim_settings.filename is not None):
                self.writer.close_file()
        return res

    def _raw_response(self,source,res,c):
        # res = np.zeros(self.pix_posy.shape)# if res is None else res
        l = np.linalg.norm(source-self.pix_pos,axis=1)
        i = np.where(l < self.pix_mod.model_size)[0]
        x = source[0]-self.pix_pos[i][:,0]
        y = source[1]-self.pix_pos[i][:,1]
        res[i] += self.pix_mod.response_spl(x,y,grid=False)*c

        # print(np.sum(res>0))
        return res
