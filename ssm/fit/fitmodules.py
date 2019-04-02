from ssm.star_cat import hipparcos
from ssm.fit.fit import FitModel, FitParameter
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time
from ctapipe.coordinates import CameraFrame, HorizonFrame
import target_calib
import numpy as np

from collections import namedtuple
from copy import deepcopy


class DataFeeder(FitModel):
    def __init__(self, data, times):
        super().__init__("FitDataFeeder")
        self.times = Time(times, format="unix")
        self.data = deepcopy(data)
        ci = []
        for k, v in self.data.items():
            ti, res = zip(*v)
            self.data[k] = (np.array(ti, dtype=np.uint64), np.array(res))
            if v[0][0] == 0:
                ci.append(k)

        self.cluster_i = ci
        self.configured = False

        self.out_data = "data"
        self.cout_times = "time_steps"
        self.cout_nframes = "n_frames"
        self.cout_clusterpix = "cluster_pixs"

    def configure(self, config):
        if not self.configured:
            config[self.cout_times] = self.times
            config[self.cout_nframes] = 1
            config[self.cout_clusterpix] = self.cluster_i

    def run(self, frame):
        frame[self.out_data] = self.data
        return frame


class TelescopeModel(FitModel):
    def __init__(
        self,
        location: EarthLocation = EarthLocation.from_geodetic(
            lon=14.974609, lat=37.693267, height=1730
        ),
        cam_config: target_calib.CameraConfiguration = None,
        focal_length: float = 2.15,
    ):
        super().__init__("Telescope")
        self.location = location
        self.cam_config = (
            cam_config
            if cam_config is not None
            else target_calib.CameraConfiguration("1.1.0")
        )
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
        self.cout_altazframes = "horiz_frames"

    def configure(self, config):
        config[self.cout_pix_pos] = self.pix_pos

        config[self.cout_fov] = 0.8  # around a pixel

        # precomputing transformations
        # Will need to do this smarter (in chunks) when the number of frames reaches 10k-100k
        self.obstimes = config["time_steps"]
        self.precomp_hf = HorizonFrame(location=self.location, obstime=self.obstimes)
        target = SkyCoord(
            ra=self.par_pointingra.val, dec=self.par_pointingdec.val, unit="deg"
        )
        self.precomp_point = target.transform_to(self.precomp_hf)
        config["start_pointing"] = self.precomp_point[0]
        config[self.cout_altazframes] = self.precomp_hf
        config["cam_frame0"] = CameraFrame(
            telescope_pointing=config["start_pointing"],
            focal_length=u.Quantity(self.focal_length, u.m),  # 28 for LST
            obstime=self.obstimes[0],
            location=self.location,
        )

    def run(self, frame):

        target = SkyCoord(
            ra=self.par_pointingra.val, dec=self.par_pointingdec.val, unit="deg"
        )

        frame[self.out_horizon_frames] = self.precomp_hf
        frame[self.out_pointings] = target.transform_to(self.precomp_hf)

        frame[self.out_cam_frames] = CameraFrame(
            telescope_pointing=frame[self.out_pointings],
            focal_length=u.Quantity(self.focal_length, u.m),  # 28 for LST
            obstime=self.obstimes,
            location=self.location,
        )
        return frame


class ProjectStarsModule(FitModel):
    def __init__(self, stars, star_coords=None, vmag_lim: tuple = (0, 9)):
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
        self.cin_cluster_pixs = "cluster_pixs"
        self.cin_pix_pos = "pix_pos"
        self.cin_altazframes = "horiz_frames"

    def configure(self, config):
        # Precomputation
        # Finding which stars are in a field of view
        # around a hotspot in the slow signal data
        self.fov = config[self.cin_fov]
        pixels = config[self.cin_cluster_pixs]
        pos = config[self.cin_pix_pos][pixels]
        pix_x = pos[:, 0] * 1e-3 * u.m
        pix_y = pos[:, 1] * 1e-3 * u.m
        s = SkyCoord(pix_x, pix_y, frame=config["cam_frame0"])
        self._get_stars_in_fov(s[0])
        config["stars_in_fov"] = self.stars[self.fov_mask]
        # Computing the alt az coordinates for the stars in the field of view
        self.star_altaz = self.star_coords[self.fov_mask].transform_to(
            config[self.cin_altazframes][:, np.newaxis]
        )

    def _get_stars_in_fov(self, pointing):
        target = pointing.transform_to("icrs")
        # stars_in_fov
        s = target.separation(self.star_coords)
        self.fov_mask = (s.deg < self.fov) & (self.stars.vmag < self.vmag_lim[1])

    def run(self, frame):
        # Transform to camera frame
        s_cam = self.star_altaz.transform_to(frame[self.in_cam_frames][:, np.newaxis])
        # Transform to cartisian coordinates in mm on the focal plane
        frame[self.out_sources] = self._StarSources(
            np.array([s_cam.x.to_value(u.mm), s_cam.y.to_value(u.mm)]).T,
            self.stars.vmag[self.fov_mask],
        )
        return frame


class IlluminationModel(FitModel):
    def __init__(self, pix_model):
        super().__init__("IlluminationModel")
        self.pix_model = pix_model
        self.in_sources = "star_sources"
        self.out_raw_response = "raw_response"
        self.in_data = "data"
        self.par_effmirrorarea = FitParameter("effmirrorarea", 6.5, [0.0, 8.0])

    def configure(self, config):
        self.lc = hipparcos.LightConverter()
        self.pix_pos = config["pix_pos"]

    def run(self, frame):
        star_srcs = frame[self.in_sources]
        data = frame[self.in_data]
        res = {}
        rates = (
            self.lc.mag2photonrate(star_srcs.vmag, area=self.par_effmirrorarea.val)
            * 1e-6
        )
        for pix, v in data.items():
            ti = v[0]
            pix_resp = np.zeros(ti.shape)  # ,ti.copy())
            ppos = self.pix_pos[pix]
            for star, rate in zip(star_srcs.pos, rates):
                l = np.linalg.norm(star[ti] - ppos, axis=1)
                i = np.where(l < self.pix_model.model_size)[0]
                x = star[:, 0][ti][i] - ppos[0]
                y = star[:, 1][ti][i] - ppos[1]
                pix_resp[i] += self.pix_model(x, y, grid=False) * rate
            res[pix] = [ti.copy(), pix_resp]
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
        for pix, v in res.items():
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
        for pix, v in inresp.items():
            respv = [v[0].copy(), None]
            respv[1] = self.calib.rate2mV(v[1])
            resp[pix] = respv
        frame[self.out_response] = resp
        return frame
