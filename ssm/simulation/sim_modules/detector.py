from ssm.core.pchain import ProcessingModule
import numpy as np
from ssdaq.core.ss_data_classes import ss_mappings
from ssm.star_cat import hipparcos


class DetectorResponse(ProcessingModule):
    def __init__(self, calib, raw_response_key="raw_response"):
        super().__init__("DetectorResponse")
        self.calib = calib
        self.raw_response_key = raw_response_key

    def configure(self, config):
        # Computing inverse mapping for full camera
        self.mapping = np.empty((32, 64), dtype=np.uint64)
        invm = np.empty(64)
        for i, k in enumerate(ss_mappings.ssl2asic_ch):
            invm[k] = i
        for i in range(32):
            self.mapping[i, :] = invm + i * 64
        self.mapping = self.mapping.flatten()

    def run(self, frame):
        frame["response"] = self.calib.rate2mV(
            frame[self.raw_response_key][self.mapping]
        )
        return frame
