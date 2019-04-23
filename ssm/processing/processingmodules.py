from ssm.core.pchain import ProcessingModule
from ssm.core.sim_io import DataReader
import numpy as np
import copy


class Reader(ProcessingModule):
    def __init__(self, filename):
        super().__init__("DataReader")

        self.reader = DataReader(filename)
        print(self.reader)

        print("Simulation config:", self.reader.sim_attr_dict)
        self.cout_camconfig = "CameraConfiguration"
        self.out_raw_resp = "raw_resp"
        self.out_time = "time"
        self._loaded_data = False

    def configure(self, config):
        # should be read from the file in the future
        from target_calib import CameraConfiguration

        config[self.cout_camconfig] = CameraConfiguration("1.1.0")
        # for the time being we load all data at once
        config["n_frames"] = 1

    def run(self, frame):
        # Only read the data once

        if not self._loaded_data:
            self.res = []
            self.times = []
            for r in self.reader.read():
                self.res.append(r.flatten())
                self.times.append(self.reader.cpu_t)
            self._loaded_data = True
        frame[self.out_raw_resp] = np.array(copy.copy(self.res))
        frame[self.out_time] = np.array(copy.copy(self.times))
        return frame


from ssm.processing.processing_utils import (
    compute_pixneighbor_map,
    find_clusters,
    get_cluster_evolution,
    smooth_slowsignal,
    evolve_clusters,
)


class SmoothSlowSignal(ProcessingModule):
    def __init__(self, n_readouts=10):
        super().__init__()
        self.n_readouts = n_readouts
        self.in_resp = "raw_resp"
        self.out_resp = "raw_resp"

    def configure(self, config):
        pass

    def run(self, frame):
        data = frame[self.in_resp]
        frame[self.out_resp] = smooth_slowsignal(data, n=self.n_readouts)
        return frame


class ClusterCleaning(ProcessingModule):
    def __init__(self, upthreshold, lothreshold):
        super().__init__("ClusterCleaning")
        self.upthreshold = upthreshold
        self.lothreshold = lothreshold
        self.cin_camconfig = "CameraConfiguration"
        self.in_data = "raw_resp"
        self.out_cleaned = "cluster_cleaned"

    def configure(self, config):
        self.pixelneighbors = compute_pixneighbor_map(config[self.cin_camconfig])

    def run(self, frame):
        data = frame[self.in_data]
        cluster_data = evolve_clusters(
            data, self.pixelneighbors, self.upthreshold, self.lothreshold
        )
        frame[self.out_cleaned] = cluster_data
        return frame
