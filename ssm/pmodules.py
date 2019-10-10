from ssm.core.sim_io import DataReader
import numpy as np
import copy
from copy import deepcopy
from astropy.coordinates import SkyCoord, EarthLocation
from ssm.core.pchain import ProcessingModule
from tqdm.auto import tqdm
from ssm.core.data import SlowSignalData
class Reader(ProcessingModule):
    def __init__(self, filename,
                 focal_length =2.15,
                 mirror_area = 6.5,
                 location= EarthLocation.from_geodetic(
            lon=14.974609, lat=37.693267, height=1730
        )
                ):
        super().__init__("DataReader")

        self.reader = DataReader(filename)
        print(self.reader)

        print("Simulation config:", self.reader.sim_attr_dict)
        self.cout_camconfig = "CameraConfiguration"
        self.out_raw_resp = "raw_resp"
        self._loaded_data = False
        self.location = location
        self.focal_length =focal_length
        self.mirror_area = mirror_area

    def configure(self, config = {}):
        # should be read from the file in the future
        from target_calib import CameraConfiguration

        self.cam_config = CameraConfiguration("1.1.0")
        self._mapping = self.cam_config.GetMapping()
        config[self.cout_camconfig] = self.cam_config
        self.pixsize = self._mapping.GetSize()
        self.pix_posx = np.array(self._mapping.GetXPixVector())
        self.pix_posy = np.array(self._mapping.GetYPixVector())
        self.pix_pos = np.array(list(zip(self.pix_posx, self.pix_posy)))
        # for the time being we load all data at once
        config["n_frames"] = 1

    def run(self, frame = {}):
        # Only read the data once

        if not self._loaded_data:
            self.res = []
            self.times = []
            for r in self.reader.read():
                self.res.append(r.flatten())
                self.times.append(self.reader.cpu_t)
            self._loaded_data = True
            self.res = np.array(self.res)
            self.times = np.array(self.times)
        frame[self.out_raw_resp] = SlowSignalData(copy.copy(self.res),
                                                  copy.copy(self.times),
                                                  {'xpix':self.pix_posx,
                                                   'ypix':self.pix_posy,
                                                    'size': self.pixsize},
                                                  focal_length=self.focal_length,
                                                  mirror_area=self.mirror_area,
                                                  location= self.location)

        return frame

from ssm.calibration.slowcal import SlowSigCal
class Calibrate(ProcessingModule):
    def __init__(self,calibration_file=None):
        super().__init__("Calibrate")
        self.in_data = 'raw_resp'
        self.out_data = "calibrated_data"
        self.cal = SlowSigCal(calibration_file)

    def configure(self,frame):
        pass
    def run(self,frame):
        data= frame[self.in_data]
        frame[self.out_data] = data.copy(self.cal.cal(data.data),data.time)
        return frame

from ssm.putils import smooth_slowsignal,find_unstable_pixs
from ssm.calibration.badpixs import get_badpixs
class PFCleaner(ProcessingModule):
    def __init__(self,):
        super().__init__("PFCleaner")
        self.badpixs = get_badpixs()
        self.in_data =  "raw_resp"
        self.out_data = "raw_resp"
        self.out_badpixs = "badpixs"
        self.out_unstablepixs = "unstablepixs"
    def configure(self,frame):
        pass
    def run(self,frame):
        data = frame[self.in_data]
        cleaned_amps = []
        cleaned_time = []


        for i,row in enumerate(data.data):

            #remove partial frames
            if np.any(np.isnan(row)):
                continue
            #mark bad pixels with nans
            row[self.badpixs] = np.nan
            cleaned_amps.append(row)
            cleaned_time.append(data.time[i])
        cleaned_amps = np.array(cleaned_amps)
        cleaned_time = np.array(cleaned_time)
        # Finally remove unstable (flickering) pixels
        unstable_pixs = find_unstable_pixs(cleaned_amps,cleaned_time)
        #cleaned_amps[:,unstable_pixs] = np.nan
        frame[self.out_data] = data.copy(cleaned_amps,cleaned_time)
        frame[self.out_badpixs] = list(self.badpixs)
        frame[self.out_unstablepixs] = list(unstable_pixs)
        return frame

class SimpleFF(ProcessingModule):
    def __init__(self,start,stop,star_thresh = 100.):
        super().__init__("SimpleFF")
        self.start_ind  = start
        self.stop_ind = stop
        self.star_thresh = star_thresh
        self.in_data =  "raw_resp"
        self.out_ff = "simple_ff"
    def configure(self,frame):
        pass
    def run(self,frame):
        #Determining FF coefficients based on first 7000 frames
        data = frame[self.in_data]
        mean_res = []
        for ii, i in enumerate(tqdm(range(self.start_ind,self.stop_ind),total=self.stop_ind-self.start_ind)):
            r = data.data[i].copy()
            r[r>self.star_thresh] = np.nan
            mean_res.append(r)
        mean_res = np.array(mean_res)
        ffc = np.nanmean(mean_res,axis=0)
        frame[self.out_ff] = ffc
        return frame


class FlatFielding(ProcessingModule):
    def __init__(self,start,stop,star_thresh = 100.):
        super().__init__("FF")
        self.start_ind  = start
        self.stop_ind = stop
        self.star_thresh = star_thresh
        self.in_data =  "raw_resp"
        self.out_ff = "ffc"
    def configure(self,frame):
        pass
    def run(self,frame):
        #Determining FF coefficients based on first 7000 frames
        data = frame[self.in_data]
        mean_res = []
        for ii, i in enumerate(tqdm(range(self.start_ind,self.stop_ind),total=self.stop_ind-self.start_ind)):
            r = data.data[i].copy()
            r[r>self.star_thresh] = np.nan
            mean_res.append(r)
        mean_res = np.array(mean_res)
        print("Mean amplitude during flat fielding",np.nanmean(mean_res))
        print(np.nanmean(mean_res,axis=0))
        ffc = np.nanmean(mean_res)/(np.nanmean(mean_res,axis=0)+0.1)
        frame[self.out_ff] = ffc
        return frame

class SmoothSlowSignal(ProcessingModule):
    def __init__(self, n_readouts=10):
        super().__init__()
        self.n_readouts = n_readouts

        self.in_data = "data"
        self.out_data = "smooth_data"
    def configure(self, config):
        pass

    def run(self, frame):
        amps = smooth_slowsignal(frame[self.in_data].data, n=self.n_readouts)
        time = frame[self.in_data].time
        data = deepcopy(frame[self.in_data])
        data.update(amps,time)
        frame[self.out_data] = data
        return frame


from ssm.processing.processing_utils import (
    compute_pixneighbor_map,
    find_clusters,
    get_cluster_evolution,
    # smooth_slowsignal,
    evolve_clusters,
)



class ClusterCleaning(ProcessingModule):
    def __init__(self, upthreshold, lothreshold):
        super().__init__("ClusterCleaning")
        self.upthreshold = upthreshold
        self.lothreshold = lothreshold
        self.in_data = "raw_resp"
        self.out_cleaned = "clusters"

    def configure(self, config):
        pass
        # self.pixelneighbors = compute_pixneighbor_map(config[self.cin_camconfig])

    def run(self, frame):
        data = frame[self.in_data]
        clusters = []
        for f in tqdm(data.data,total=len(data.data)):
            std = np.nanstd(f)
            mean = np.nanmean(f)
            clusters.append(find_clusters(f,mean +self.upthreshold*std, mean +self.lothreshold*std, data.neighbors))
        # cluster_data = evolve_clusters(
        #     data.data, data.neighbors, self.upthreshold, self.lothreshold
        # )
        frame[self.out_cleaned] = clusters#_data
        return frame
