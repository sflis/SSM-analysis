import numpy as np
from tqdm.auto import tqdm
def compute_pixneighbor_map(xpix,
                            ypix,
                            size,
                            pixdist: float = 1.8,
                            intertmdist: float = 2.5
) -> list:
    """ Computes a pixel neighbor map for given pixel distances
        within an TM and between pixels at adjecent TMs.
    """
    dist = pixdist * size
    intdist = intertmdist * size
    neighbors = []  # [[]]*2048
    for i in range(2048):
        neighbors.append([])
    for i in tqdm(range(2048), total=2048, desc="Mapping neighbor pixels"):
        for j in range(i + 1, 2048):

            dx = xpix[i] - xpix[j]
            dy = ypix[i] - ypix[j]
            tm1 = int(i / 64)
            tm2 = int(j / 64)
            d = dist if tm1 == tm2 else intdist
            if np.sqrt(dx ** 2 + dy ** 2) < (d):
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


class SlowSignalData:
    _neighbors = None
    def __init__(self,
                 data,
                 time,
                 mapping = None,
                 focal_length = None,
                 mirror_area = None,
                 location=None):

        self.data = data
        self._time = time
        if data.shape[0] != time.shape[0]:
            raise ValueError(
                    "Number of samples ({}) must match number of timestamps ({})".format(data.shape[0],
                                                           time.shape[0]) )
        self._xpix = mapping['xpix']
        self._ypix = mapping['ypix']
        self.size = mapping['size']
        self.pix_pos = np.array(list(zip(self.xpix, self.ypix)))
        self._loc = location
        self.focal_length = focal_length
        self.mirror_area = mirror_area
        SlowSignalData._neighbors = self.neighbors = SlowSignalData._neighbors or compute_pixneighbor_map(self.xpix,
                            self.ypix,
                            self.size,)

    @property
    def time(self):
        return self._time
    @property
    def xpix(self):
        return self._xpix
    @property
    def ypix(self):
        return self._ypix
    @property
    def location(self):
        return self._loc

    def update(self, data,time):
        self.data = data
        self._time = time

    def copy(self,data,time):
        cp = SlowSignalData(data,
                            time,
                            {"xpix":self.xpix,"ypix":self.ypix,"size":self.size},
                            self.focal_length,
                            self.mirror_area,
                            self.location,
                           )
        return cp