from ssm.core.pchain import ProcessingModule
from ssm.core.sim_io import SimDataWriter, SourceDescr

from ssdaq.data import SSReadout
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
        self.timestamp_key = time_key
        self.in_time_key = "time"

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
        cputime = frame[self.in_time_key].unix
        s = int(cputime)
        ns = int((cputime - s) * 1e9)
        self.writer.write_readout(
            SSReadout(
                readout_number=frame[self.iro_key],
                timestamp=frame[self.timestamp_key],
                cpu_t_s=s,
                cpu_t_ns=ns,
                data=frame[self.response_key].reshape((32, 64)),
            ),
            frame["star_sources"].pos,
        )
        p = frame["tel_pointing"].transform_to("icrs")
        self.writer.write_tel_data(p.ra.deg, p.dec.deg, cputime, s, ns)

    def finish(self, config):
        self.writer.close_file()
