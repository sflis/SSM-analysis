import pickle
import numpy as np
from ssm.utils.remotedata import load_data_factory
import ssm
import os

root_dir = os.path.dirname(os.path.dirname(ssm.__file__))
defaultcal = load_data_factory(
    "https://owncloud.cta-observatory.org/index.php/s/HuUSjikMcrKqV6t/download",
    os.path.join(root_dir, "caldata", "defaultslowcal.pkl"),
    "slowcal",
    lambda x: x,
)


class SlowSigCal:

    """Contains calibration parameters for the slow signal path.
        The amplitude calibration is modeled with a 2nd degree polynomial as a
        function of light intensity:
            y = ax^2+bx+c
        where y is the amplitude and x is the light intensity. The inverse of this
        function is evaluated when applying the calibration.


    Attributes:
        a (array): a parameters for all pixels
        b (array): b parameters for all pixels
        badpixs (dict): dictionary containging sets of bad pixels
        c (array): b parameters for all pixels
        params (dict): dict with parameters per pixel
    """

    def __init__(self, file=None):
        file = file or defaultcal()
        cal = pickle.load(open(file, "rb"))
        self.badpixs = cal["badpixs"]
        self.params = cal["cal"]
        self.a = np.ones(2048) * np.nan
        self.b = np.ones(2048) * np.nan
        self.c = np.ones(2048) * np.nan
        for pix, p in self.params.items():
            if pix in self.badpixs["unphysical_cal"]:
                continue
            self.a[pix] = p[0][0]
            self.b[pix] = p[0][1]
            self.c[pix] = p[0][2]

    def cal(self, data: np.array) -> np.array:
        """ Applies slow signal calibration to raw data.

            Pixels that have an unphysical calibration are set to NaN
        Args:
            data (np.array): raw data

        Returns:
            np.array: calibrated data
        """
        data = data.copy()
        data[data == 0] = np.nan
        return -(self.b - np.sqrt(self.b * self.b - 4 * self.a * (self.c - data))) / (
            2 * self.a
        )
        # (-b + sqrt(b^2-4ac))/(2a)
