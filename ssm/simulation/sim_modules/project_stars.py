from astropy.coordinates import SkyCoord
import astropy.units as u
from collections import namedtuple

from ssm.core.pchain import ProcessingModule
import numpy as np

class ProjectStars(ProcessingModule):
    def __init__(self,stars,vmag_lim = (0,9)):
        super().__init__('ProjectStars')
        self.stars = stars
        self.stars_in_fov = None
        self.fov_star_mask = None
        self.star_coords = SkyCoord(ra=stars.ra_deg.values,dec=stars.dec_deg.values,unit='deg')
        self.vmag_lim = vmag_lim
        self._StarSources = namedtuple("StarSources",'pos vmag')

    def configure(self,config):
        self.fov = config['fov']
        self._get_stars_in_fov(config['start_pointing'])
        config['stars_in_fov'] = self.stars[self.fov_mask]

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