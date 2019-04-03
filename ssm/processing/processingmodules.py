from ssm.core.pchain import ProcessingModule
from ssm.core.sim_io import DataReader


class Reader(ProcessingModule):
    def __init__(self, filename):
        super().__init__("DataReader")

        self.reader = DataReader(filename)
        print(self.reader)

        print("Simulation config:", self.reader.sim_attr_dict)
        self.cout_camconfig = "CameraConfiguration"
        self.out_raw_resp = "raw_resp"
        self.out_time = "time"

    def configure(self, config):
        # should be read from the file in the future
        from target_calib import CameraConfiguration

        config[self.cout_camconfig] = CameraConfiguration("1.1.0")
        #for the time being we load all data at once
        config["n_frames"] = 1

    def run(self, frame):
        res = []
        times = []
        import copy

        for r in self.reader.read():
            res.append(r.flatten())
            times.append(self.reader.cpu_t)
        frame[self.out_raw_resp] = res
        frame[self.out_time] = times
        return frame


from ssm.processing.processing_utils import (
    compute_pixneighbor_map,
    find_clusters,
    get_cluster_evolution,
)


class ClusterCleaning(ProcessingModule):
    def __init__(self, upthreshold, lothreshold):
        super().__init__("ClusterCleaning")
        self.upthreshold = upthreshold
        self.lothreshold = lothreshold
        self.cin_camconfig = "CameraConfiguration"
        self.in_data = 'raw_resp'
        self.out_cleaned = 'cluster_cleaned'
    def configure(self, config):
        self.pixelneighbors = compute_pixneighbor_map(config[self.cin_camconfig])

    def run(self, frame):
        data = frame[self.in_data]
        clusters = find_clusters(data[0], self.upthreshold, self.lothreshold, self.pixelneighbors)
        cluster_data, ipixs = get_cluster_evolution(
            clusters, data, self.pixelneighbors, self.upthreshold, self.lothreshold
        )
        frame[self.out_cleaned] = cluster_data
        return frame
