import numpy as np
from datetime  import datetime
badpix_maps = {datetime(2019,5,6):np.array([  25,   58,  101,  304,  449,  570,  653, 1049, 1094, 1158, 1177,
       1381, 1427, 1434, 1439, 1765, 1829, 1869, 1945, 1957, 2009, 2043])}

def get_badpixs():
    # now = datetime.now()
    keys = sorted(badpix_maps.keys())
    return badpix_maps[keys[-1]]