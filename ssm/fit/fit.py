from ssm.core.pchain import ProcessingModule, ProcessingChain
from ssm.core.util_pmodules import Aggregate
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time
from ctapipe.coordinates import CameraFrame, HorizonFrame
import target_calib
import numpy as np
_c = target_calib.CameraConfiguration("1.1.0")
_mapping = _c.GetMapping()

class PointingFit(ProcessingModule):
    def __init__(self, data, times,cluster_i):
        super().__init__("FitStarter")
        self.times = Time(times, format="unix")
        self.data = np.swapaxes(np.array(data),0,1)
        self.cluster_i = cluster_i
        self.fit_chain = ProcessingChain(silent=False)
        self.fit_chain.add(self)
        self.in_exp = "response"
        self.configured = False
        self.out_cluster_i = "cluster_i"
        self.out_data = "data"
        self.params = []
    def addFitModule(self, module):
        self.params += module.parameters.values()
        self.fit_chain.add(module)

    def configure(self, config):
        if not self.configured:
            config["time_steps"] = self.times
            config["n_frames"] = 1
            config["cluster_i0"] = self.cluster_i[0]
            config["cluster_i"] = self.cluster_i
            self.aggregator = Aggregate([self.in_exp])
            self.fit_chain.add(self.aggregator)

    def run(self, frame):
        frame[self.out_data] = self.data
        frame[self.out_cluster_i] = self.cluster_i
        return frame

    def compute_exp(self, x = None):
        if(x is not None):
            for i,xp in enumerate(x):
                self.params[i].val = xp
        self.fit_chain.run()
        self.fit_chain.config_run = True
        exp = self.aggregator.aggr[self.in_exp][0]
        self.aggregator.clear()
        return exp

    def __call__(self, x):
        print(x)
        exp = self.compute_exp( x)
        m = self.data>0
        chi2 = np.sum((exp[m] - self.data[m]) ** 2/self.data[m])
        print(chi2)
        return chi2


class FitModel(ProcessingModule):
    def __init__(self, name):
        super().__init__(name)
        self._par = {}
        self._par_registered = False

    def _registerparams(self):
        if not self._par_registered:
            # Introspecting to find all input parameters that should be
            # changed to properties
            for iok, iov in self.__dict__.items():
                if iok[:4] == "par_":
                    self._par[iok[4:]] = iov
                    setattr(
                        self.__class__,
                        iok,
                        property(
                            lambda self, k=iok[4:]: self._par[k],
                            lambda self, v, k=iok[4:]: self._par.update({k: v}),
                        ),
                    )
        self._par_registered = True

    @property
    def parameters(self):
        if not self._par_registered:
            self._registerparams()
        return self._par


class FitParameter:
    def __init__(self, name, val0, interval=None):
        self.name = name
        self.val = val0
        self.interval = interval
    def __repr__(self):
        return "{}: {} [{},{}]".format(self.name,self.val,*self.interval)

class TelescopeModel(FitModel):
    def __init__(
        self,
        location=EarthLocation.from_geodetic(lon=14.974609, lat=37.693267, height=1730),
        cam_config=_c,
        focal_length=2.15,
    ):
        super().__init__("Telescope")
        self.location = location
        self.cam_config = cam_config
        self.focal_length = focal_length
        self._mapping = self.cam_config.GetMapping()
        self.pix_posx = np.array(self._mapping.GetXPixVector()) * 1e3  # mm
        self.pix_posy = np.array(self._mapping.GetYPixVector()) * 1e3  # mm
        self.pix_pos = np.array(list(zip(self.pix_posx, self.pix_posy)))
        self.par_pointingra = FitParameter("pointingra", 0.0, [-10.0, 370.0])
        self.par_pointingdec = FitParameter("pointingdec", 0.0, [-90.0, 90.0])

        self.out_horizon_frames = "horizon_frames"
        self.out_pointings = "pointings"
        self.out_cam_frames = "cam_frames"
        self.cout_pix_pos = "pix_pos"
        self.cout_fov = "fov"

    def configure(self, config):
        config[self.cout_pix_pos] = self.pix_pos

        config[self.cout_fov] = 0.8  # around a pixel

        # precomputing transformations
        # Will need to do this smarter (in chunks) when the number of frames reaches 10k-100k
        self.obstimes = config["time_steps"]
        self.precomp_hf = HorizonFrame(location=self.location, obstime=self.obstimes)
        target = SkyCoord(ra=self.par_pointingra.val, dec=self.par_pointingdec.val, unit="deg")
        self.precomp_point = target.transform_to(self.precomp_hf)
        config["start_pointing"] = self.precomp_point[0]
        config["cam_frame0"] = CameraFrame(
            telescope_pointing=config["start_pointing"],
            focal_length=u.Quantity(self.focal_length, u.m),  # 28 for LST
            obstime=self.obstimes[0],
            location=self.location,
        )

    def run(self, frame):

        target = SkyCoord(ra=self.par_pointingra.val, dec=self.par_pointingdec.val, unit="deg")

        frame[self.out_horizon_frames] = self.precomp_hf
        frame[self.out_pointings] = target.transform_to(self.precomp_hf)

        frame[self.out_cam_frames] = CameraFrame(
            telescope_pointing=frame[self.out_pointings],
            focal_length=u.Quantity(self.focal_length, u.m),  # 28 for LST
            obstime=self.obstimes,
            location=self.location,
        )
        return frame


from collections import namedtuple


class ProjectStarsModule(FitModel):
    def __init__(self, stars, star_coords=None, vmag_lim=(0, 9)):
        super().__init__("ProjectStars")
        self.stars = stars
        self.stars_in_fov = None
        self.fov_star_mask = None
        self.star_coords = (
            SkyCoord(ra=stars.ra_deg.values, dec=stars.dec_deg.values, unit="deg")
            if star_coords is None
            else star_coords
        )
        self.vmag_lim = vmag_lim
        self._StarSources = namedtuple("StarSources", "pos vmag")
        self.in_cam_frames = "cam_frames"
        self.out_sources = "star_sources"
        self.cin_fov = "fov"
        self.cin_cluster_i = "cluster_i0"
        self.cin_pix_pos = "pix_pos"

    def configure(self, config):
        self.fov = config[self.cin_fov]
        pixels = config[self.cin_cluster_i]
        pos = config[self.cin_pix_pos][pixels]
        pix_x = pos[:, 0] * 1e-3 * u.m
        pix_y = pos[:, 1] * 1e-3 * u.m
        s = SkyCoord(pix_x, pix_y, frame=config["cam_frame0"])
        self._get_stars_in_fov(s[0])
        config["stars_in_fov"] = self.stars[self.fov_mask]

    def _get_stars_in_fov(self, pointing):
        target = pointing.transform_to("icrs")
        # stars_in_fov
        s = target.separation(self.star_coords)
        self.fov_mask = (s.deg < self.fov) & (self.stars.vmag < self.vmag_lim[1])

    def run(self, frame):
        s_cam = self.star_coords[self.fov_mask].transform_to(
            frame[self.in_cam_frames][:, np.newaxis]
        )

        frame[self.out_sources] = self._StarSources(
            np.array([s_cam.x.to_value(u.mm), s_cam.y.to_value(u.mm)]).T,
            self.stars.vmag[self.fov_mask],
        )
        return frame

from ssm.star_cat import hipparcos
class IlluminationModel(FitModel):
    def __init__(self, pix_model):
        super().__init__("IlluminationModel")
        self.pix_model = pix_model
        self.in_sources = "star_sources"
        self.out_raw_response = "raw_response"
        self.par_effmirrorarea = FitParameter("effmirrorarea", 6.5, [0.0, 8.0])
        self.cin_cluster_i = 'cluster_i'
    def configure(self, config):
        self.lc = hipparcos.LightConverter()
        self.pix_pos = config["pix_pos"]
        self.cluster_i = config[self.cin_cluster_i]
    def run(self, frame):
        star_srcs = frame[self.in_sources]
        res = np.zeros((2048, star_srcs.pos.shape[1]))
        rates = self.lc.mag2photonrate(star_srcs.vmag, area=self.par_effmirrorarea.val)* 1e-6
        for star,rate in zip(star_srcs.pos,rates):
            for spos,r,ci in zip(star,np.swapaxes(res,0,1),self.cluster_i):
                # self._raw_response(spos, r, rate)
                # ind[ci] = 1
                self._raw_response2(spos, r, ci,rate)
                # ind = np.arange(2048)
                # ind[ci] = -1
                # ind = ind[ind>0]
                # r[ind] = np.nan

        res[res<0.001] = np.nan
        frame[self.out_raw_response] = res
        return frame

    def _raw_response(self, source, res, rate):
        l = np.linalg.norm(source - self.pix_pos, axis=1)
        i = np.where(l < self.pix_model.model_size)[0]
        x = source[0] - self.pix_pos[i][:, 0]
        y = source[1] - self.pix_pos[i][:, 1]
        res[i] += self.pix_model(x, y, grid=False) * rate

    def _raw_response2(self, source, res,m, rate):
        l = np.linalg.norm(source - self.pix_pos, axis=1)
        i = np.where(l < self.pix_model.model_size)[0]
        ind = list(set(i).intersection(m))
        x = source[0] - self.pix_pos[ind][:, 0]
        y = source[1] - self.pix_pos[ind][:, 1]
        res[ind] += self.pix_model(x, y, grid=False) * rate

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
        res[res > 0] += self.par_nsbrate.val
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
        frame[self.out_response] = self.calib.rate2mV(frame[self.in_raw_response])

        return frame
