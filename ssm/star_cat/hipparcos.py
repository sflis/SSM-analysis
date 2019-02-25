import pandas as pd
from os import path
import ssm
from astropy.coordinates import SkyCoord
import numpy as np
# import astropy.units as u

def load_hipparcos_cat():
    ssm_path = path.dirname(ssm.__file__)
    cat_file = path.join(ssm_path,'resources/HipparcosCatalog_lt9.txt')
    #Need to parse the header manually to get col names correctly
    header = open(cat_file,'r')
    header.readline()
    header.readline()
    l = header.readline()
    ls = list(map(lambda x:x.strip(),l.split('|')))

    dt = pd.read_csv(cat_file,delimiter = '|',header=2,usecols=np.arange(1,14),names=ls)
    #sanitize cataloge
    m = (pd.to_numeric(dt.ra_deg,errors='coerce')>0 )
    stars = SkyCoord(ra=dt.ra_deg[m].values,dec=dt.dec_deg[m].values,unit='deg')
    sel = dt[m]
    return sel,stars