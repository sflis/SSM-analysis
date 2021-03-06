from ssm.core.pchain import ProcessingModule

from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

from ctapipe.coordinates import EngineeringCameraFrame,CameraFrame, AltAz  # HorizonFrame

import numpy as np


class Telescope(ProcessingModule):
    def __init__(
        self,
        location=EarthLocation.from_geodetic(lon=14.974622, lat=37.693266, height=1730),
        cam_config=None,
        focal_length=2.15,
        target=None,
        fov=5,
    ):
        super().__init__("Telescope")
        self.location = location
        import target_calib

        self.cam_config = (
            target_calib.CameraConfiguration("1.1.0")
            if cam_config is None
            else cam_config
        )
        self.focal_length = focal_length
        self._mapping = self.cam_config.GetMapping()
        self.pix_posx = np.array(self._mapping.GetXPixVector()) * 1e3  # mm
        self.pix_posy = np.array(self._mapping.GetYPixVector()) * 1e3  # mm
        self.pix_pos = np.array(list(zip(self.pix_posx, self.pix_posy)))
        self.target = target
        self.fov = fov

    def configure(self, config):
        config["pix_pos"] = self.pix_pos
        config["fov"] = self.fov

        td = config["time_step"]
        n_frames = config["n_frames"]

        # precomputing transformations
        # Will need to do this smarter (in chunks) when the number of frames reaches 10k-100k
        self.obstimes = np.array(
            [config["start_time"] + td * i for i in range(n_frames)]
        )
        self.precomp_hf = AltAz(location=self.location, obstime=self.obstimes)
        if isinstance(self.target, SkyCoord):
            self.precomp_point = self.target.transform_to(self.precomp_hf)
        else:
            alt, az = self.target
            n = len(self.obstimes)
            self.precomp_point = SkyCoord(alt=alt*np.ones(n), az=az*np.ones(n), frame=self.precomp_hf,)

        self.precomp_camf = EngineeringCameraFrame(
            n_mirrors=2,
            telescope_pointing=self.precomp_point,
            focal_length=u.Quantity(self.focal_length, u.m),  # 28 for LST
            obstime=self.obstimes,
            location=self.location,
        )
        config["start_pointing"] = self.precomp_point[0]

    def run(self, frame):
        obstime = frame["time"]
        horizon_frame = self.precomp_hf[frame["frame_n"] - 1]
        pointing = self.precomp_point[frame["frame_n"] - 1]
        current_cam_frame = self.precomp_camf[frame["frame_n"] - 1]
        frame["horizon_frame"] = horizon_frame
        frame["pointing"] = pointing
        frame["tel_pointing"] = pointing
        frame["current_cam_frame"] = current_cam_frame
        return frame
