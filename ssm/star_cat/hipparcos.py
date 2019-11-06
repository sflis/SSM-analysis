import pandas as pd
from os import path
import ssm
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u

# Photon Flux:
# Given the passband and the magnitude of an object, the number of tphotons incident at the top of the atmosphere may be estimated using the data in this table:
# Band    lambda_c    dlambda/lambda  Flux at m=0 Reference
# &mu     Jy
# U   0.36    0.15    1810    Bessel (1979)
# B   0.44    0.22    4260    Bessel (1979)
# V   0.55    0.16    3640    Bessel (1979)
# R   0.64    0.23    3080    Bessel (1979)
# I   0.79    0.19    2550    Bessel (1979)
# J   1.26    0.16    1600    Campins, Reike, & Lebovsky (1985)
# H   1.60    0.23    1080    Campins, Reike, & Lebovsky (1985)
# K   2.22    0.23    670 Campins, Reike, & Lebovsky (1985)
# g   0.52    0.14    3730    Schneider, Gunn, & Hoessel (1983)
# r   0.67    0.14    4490    Schneider, Gunn, & Hoessel (1983)
# i   0.79    0.16    4760    Schneider, Gunn, & Hoessel (1983)
# z   0.91    0.13    4810    Schneider, Gunn, & Hoessel (1983)
# Also useful are these identities:

# 1 Jy = 10^-23 erg sec^-1 cm^-2 Hz^-1
# 1 Jy = 1.51e7 photons sec^-1 m^-2 (dlambda/lambda)^-1
# Example: How many V-band photons are incident per second on an area of 1 m^2 at the top of the atmosphere from a V=23.90 star? From the table, the flux at V=0 is 3640 Jy; hence, at V=23.90 the flux is diminished by a factor 10^(-0.4*V)=2.75e-10, yielding a flux of 1.e-6 Jy. Since dlambda/lambda=0.16 in V, the flux per second on a 1 m^2 aperture is
# f=1.e-6 Jy * 1.51e7 * 0.16 = 2.42 photons sec^-1


class LightConverter:
    def __init__(self):
        self.photon_f = {
            "V": (0.55, 0.16, 3640),
            "B": (0.44, 0.22, 4260),
            "U": (0.36, 0.15, 1810),
        }

    def mag2dflux(self, mag, band="V"):
        return 10 ** (-0.4 * mag) * self.photon_f[band][2]

    def mag2photonrate(self, mag, band="V", area=6.5):
        return (
            10 ** (-0.4 * mag)
            * self.photon_f[band][2]
            * self.photon_f[band][1]
            * area
            * 1.51e7
        )


import pickle


def load_hipparcos_cat():
    root_dir = path.dirname(path.dirname(ssm.__file__))
    computed_file = path.join(root_dir, "caldata", "hip_catalog_computed.pkl")
    if path.exists(computed_file):
        with open(computed_file, "rb") as f:
            data = pickle.load(f)
            return data["sel"], data["stars"]

    ssm_path = path.dirname(ssm.__file__)
    cat_file = path.join(ssm_path, "resources/HipparcosCatalog_lt9.txt")

    # Need to parse the header manually to get col names correctly
    header = open(cat_file, "r")
    header.readline()
    header.readline()
    l = header.readline()
    ls = list(map(lambda x: x.strip(), l.split("|")))
    # Naming the unnamed columns so that read_csv does not compain
    ls[0] = "1"
    ls[-1] = "2"

    dt = pd.read_csv(
        cat_file, delimiter="|", header=2, usecols=np.arange(1, 14), names=ls
    )
    # sanitize cataloge
    m = pd.to_numeric(dt.ra_deg, errors="coerce") > 0
    stars = SkyCoord(
        ra=dt.ra_deg[m].values, dec=dt.dec_deg[m].values, unit="deg", frame="icrs"
    )
    sel = dt[m]
    with open(computed_file, "wb") as f:
        data = pickle.dump(dict(sel=sel, stars=stars), f)
    return sel, stars
