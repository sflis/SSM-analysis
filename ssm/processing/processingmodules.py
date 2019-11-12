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
        self.in_time = "time"
        self.out_time = "time"

    def configure(self, config):
        pass

    def run(self, frame):
        frame[self.out_resp] = smooth_slowsignal(frame[self.in_resp], n=self.n_readouts)
        frame[self.out_time] = frame[self.in_time][self.n_readouts - 1 :]
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


from collections import namedtuple
from scipy.interpolate import UnivariateSpline


class ClusterReduction(ProcessingModule):
    def __init__(self, s=1e4, reduction=10, name=None):
        super().__init__(name)
        self.s = s
        self.in_data = "cluster_cleaned"
        self.in_time = "time"
        self.out_cluster_spl = "cluster_spl"
        self.out_data = "cluster_cleaned"
        self.out_time = "time"
        self.reduction = reduction

    def configure(self, config):
        pass

    def run(self, frame):
        clusters = frame[self.in_data]
        time = frame[self.in_time]
        splclusters = []
        reduced_clusters = []

        # Flat
        rtime = time[:: self.reduction]
        rtind = np.arange(0, time.shape[0], self.reduction)
        revtind = np.zeros(time.shape)
        revtind[rtind] = np.arange(0, len(rtind))

        for cluster in clusters:
            splclusterdata = {}
            clusterdata = {}
            d1 = []
            for pix, data in sorted(cluster.items()):
                data = np.array(data)
                indices = np.array(data[:, 0], dtype=np.uint64)

                splclusterdata[pix] = UnivariateSpline(
                    time[indices], data[:, 1], s=self.s
                )
                smoothspl = UnivariateSpline(time[indices], data[:, 1], s=5e6)
                spld1 = smoothspl.derivative(n=1)
                d1.append(np.max(np.abs(spld1(time[indices]))))
                inters, rindices, r = np.intersect1d(
                    rtind, indices, assume_unique=True, return_indices=True
                )  # [::self.reduction]
                # print(rtind)
                # print(indices)
                # rindices = revtind[rindices]
                print(np.max(rindices))
                clusterdata[pix] = list(
                    zip(rindices, splclusterdata[pix](rtime[rindices]))
                )
            d1 = np.array(d1)
            print(d1, len(d1), np.sum(d1 > 1.0))
            # We want a least two pixels participating in
            # a cluster with at least two of the pixels having
            # a derivate of more than 1
            if len(d1) > 1 and (np.sum(d1 > 1.0) > 1):
                splclusters.append(splclusterdata)
                reduced_clusters.append(clusterdata)
        frame[self.out_data] = reduced_clusters
        frame[self.out_cluster_spl] = splclusters
        frame[self.out_time] = rtime
        return frame
