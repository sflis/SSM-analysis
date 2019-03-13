from ssm.core.pchain import ProcessingModule
from ssm.core.sim_io import SimDataWriter, SourceDescr

from ssdaq import SSReadout
import numpy as np


class Writer(ProcessingModule):
    def __init__(
        self,
        filename,
        response_key="response",
        iro_key="frame_n",
        time_key="timestamp",
        name="DataWriter",
    ):
        super().__init__(name)
        self.filename = filename
        self.response_key = response_key
        self.iro_key = iro_key
        self.time_key = time_key

    def configure(self, config):
        srcs = []
        sim_attrs = {"sim_config": {"modules": np.array(config["modules"])}}
        for ss in config["stars_in_fov"].iterrows():
            s = ss[1]
            srcs.append(
                SourceDescr(s.name, float(s.ra_deg), float(s.dec_deg), float(s.vmag))
            )

        self.writer = SimDataWriter(
            self.filename, sim_sources=srcs, sim_attrs=sim_attrs, buffer=1000
        )

    def run(self, frame):

        self.writer.write_readout(
            SSReadout(
                readout_number=frame[self.iro_key],
                timestamp=frame[self.time_key],
                data=frame[self.response_key].reshape((32, 64)),
            ),
            frame["star_sources"].pos,
        )
        p = frame["tel_pointing"].transform_to("icrs")
        time = frame[self.time_key]
        s = int(time * 1e-9)
        ns = int(time - s)
        self.writer.write_tel_data(p.ra.deg, p.dec.deg, time, s, ns)

    def finish(self, config):
        self.writer.close_file()
