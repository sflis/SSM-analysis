from __future__ import generator_stop
from ssdaq.data.io import SSDataWriter, SSDataReader
from collections import namedtuple
from tables import IsDescription, UInt64Col, Float32Col, Atom, NaturalNameWarning
import tables
import numpy as np


class SimSSTableDs(IsDescription):

    iro = UInt64Col()
    ro_time = UInt64Col()
    src_pos = Float32Col((2))


SourceDescr = namedtuple("SourceDescr", "name ra dec vMag")


class SimDataWriter(SSDataWriter):
    """A writer for Slow Signal simulation data"""

    def __init__(
        self,
        filename,
        attrs=None,
        filters=None,
        sim_sources=None,
        sim_attrs=None,
        buffer=1,
    ):
        super().__init__(filename, attrs, filters, buffer)
        self.simgroup = self.file.create_group(
            self.file.root, "SlowSignalSimulation", "Slow signal simulation data"
        )
        self.sim_tables = []
        if sim_sources is not None:
            for i, source in enumerate(sim_sources):
                table = self.file.create_table(
                    self.simgroup,
                    "sim_source%d" % i,
                    SimSSTableDs,
                    "Simulation data for %s" % source.name,
                )
                self.sim_tables.append((table, table.row))
                self.tables.append(table)  # For flush optimization
                table.attrs["name"] = source.name
                table.attrs["ra"] = source.ra
                table.attrs["dec"] = source.dec
                table.attrs["vMag"] = source.vMag

        self.table.attrs["simulation"] = True
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", tables.NaturalNameWarning)
            warnings.simplefilter("ignore", tables.FlavorWarning)
            if sim_attrs is not None:
                for k, sa in sim_attrs.items():
                    for sub_k, sub_sa in sa.items():
                        atom = Atom.from_dtype(sub_sa.dtype)
                        ds = self.file.create_carray(
                            self.simgroup, "{}.{}".format(k, sub_k), atom, sub_sa.shape
                        )
                        ds[:] = sub_sa

    def write_readout(self, ro, sim=None):
        super().write_readout(ro)
        if sim is not None:
            for i, s in enumerate(sim):
                row = self.sim_tables[i][1]
                row["iro"] = ro.iro
                row["ro_time"] = ro.time
                row["src_pos"] = s
                row.append()


class DataReader(SSDataReader):
    """A reader for Slow Signal data"""

    def __init__(self, filename, mapping="ssl2asic_ch"):
        """

            Args:
                filename (str): path to file

            Kwargs:
                mapping (str): determines how the pixels are mapped. Two mappings
                               are availabe: 'ssl2colrow' and 'ssl2asic_ch' which correspond to
                               a 2D col-row layout and ASIC-channel which is the same ordering used for the
                               fast signal data.


        """
        super().__init__(filename, mapping)
        self.data_type = "Raw"
        self._source_pos = None
        self.sim_tables = []
        self.sim_attr_dict = {}

        if "simulation" in self.attrs:
            self.data_type = "Simulation"

            for st in self.file.root.SlowSignalSimulation:
                if st.attrs.CLASS == "TABLE":
                    self.sim_tables.append(st)

                if st.attrs.CLASS == "CARRAY":
                    base, sub = tuple(st.name.split("."))
                    if base not in self.sim_attr_dict:
                        self.sim_attr_dict[base] = {}
                    self.sim_attr_dict[base][sub] = st[:]

            # putting the simulation attributes into a namedtuple of namedtuples
            SimAttrs = namedtuple("SimAttrs", " ".join(list(self.sim_attr_dict.keys())))
            sattrs = []
            for k, v in self.sim_attr_dict.items():
                SimAttr = namedtuple(k, " ".join(v.keys()))
                fields = []
                for k_field, field in v.items():
                    fields.append(field)
                sattrs.append(SimAttr(*fields))

            self.sim_attrs = SimAttrs(*sattrs)
            self._source_pos = np.zeros((len(self.sim_tables), 2))

    @property
    def source_pos(self):
        return self._source_pos.copy()

    def read(self, start=None, stop=None, step=None):

        if stop is None and start is not None:
            stop = start + 1
        sim_iters = []
        for st in self.sim_tables:
            sim_iters.append(st.iterrows(start, stop, step))
        for r in self._read(start, stop, step):
            for i, si in enumerate(sim_iters):
                sr = si.__next__()
                # print(i,sr['iro'])
                self._source_pos[i, :] = sr["src_pos"]
            yield r

    def __str__(self):
        s = "SSDataReader:\n"
        s += "    Filename:%s\n" % self.filename
        s += "    Title: CHEC-S Slow signal monitor data\n"
        s += "    n_readouts: %d\n" % self.n_readouts
        s += "    ssdata-version: %d\n" % self.attrs.ss_data_version
        s += "    created with ssdaq version: %s\n" % self.attrs.ssdaq_version
        s += "    data_type: %s\n" % self.data_type
        if self.data_type == "Simulation":
            s += "    number of simulated sources: %s\n" % len(self.sim_tables)

        return s
