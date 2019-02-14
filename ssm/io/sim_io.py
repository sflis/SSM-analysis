from __future__ import generator_stop
from ssdaq import SSDataWriter, SSDataReader
from collections import namedtuple
from tables import IsDescription,UInt64Col,Float32Col
import numpy as np

class SimSSTableDs(IsDescription):

    readout_number = UInt64Col()
    ro_time        = UInt64Col()
    src_pos        = Float32Col((2))


SourceDescr = namedtuple('SourceDescr','name ra dec vMag')


class SimDataWriter(SSDataWriter):
    """A writer for Slow Signal simulation data"""
    def __init__(self,filename, attrs = None,filters = None,sim_sources=None):
        super().__init__(filename,attrs,filters)
        self.simgroup = self.file.create_group(self.file.root, 'SlowSignalSimulation', 'Slow signal simulation data')
        self.sim_tables = []
        if(sim_sources is not None):
            for i, source in enumerate(sim_sources):
                table = self.file.create_table(self.simgroup, 'sim_source%d'%i, SimSSTableDs, "Simulation data for %s"%source.name)
                self.sim_tables.append((table,table.row))
                table.attrs['name'] = source.name
                table.attrs['ra']   = source.ra
                table.attrs['dec']   = source.dec
                table.attrs['vMag'] = source.vMag

        self.table.attrs['simulation'] =True


    def write_readout(self,ro,sim=None):
        super().write_readout(ro)
        if(sim is not None):
            for i,s in enumerate(sim):
                row = self.sim_tables[i][1]
                row['readout_number'] = ro.readout_number
                row['ro_time'] = ro.readout_timestamp
                row['src_pos'] = s
                row.append()
                self.sim_tables[i][0].flush()

    def close_file(self):
        '''Closes file handle
        '''
        for t in self.sim_tables:
            t[0].flush()
        super().close_file()




class DataReader(SSDataReader):
    """A reader for Slow Signal data"""
    def __init__(self, filename, mapping='ssl2asic_ch'):
        """

            Args:
                filename (str): path to file

            Kwargs:
                mapping (str): determines how the pixels are mapped. Two mappings
                               are availabe: 'ssl2colrow' and 'ssl2asic_ch' which correspond to
                               a 2D col-row layout and ASIC-channel which is the same ordering used for the
                               fast signal data.


        """
        super().__init__(filename,mapping)
        self.data_type = 'Raw'
        self.source_pos = None
        for group in self.file.walk_groups():
            print(group)
        self.sim_tables = []
        if('simulation' in self.attrs):
            self.data_type = 'Simulation'
            for st in self.file.root.SlowSignalSimulation:
                self.sim_tables.append(st)
            self.source_pos = np.zeros((len(self.sim_tables),2))
    def read(self,start=None,stop=None,step=None):

        if(stop is None and start is not None):
            stop = start+1
        sim_iters = []
        for st in self.sim_tables:
            sim_iters.append(st.iterrows(start,stop,step))
        for r in self._read(start,stop,step):
            for i,si in enumerate(sim_iters):
                sr = si.__next__()
                self.source_pos[i,:] =sr['src_pos']
            yield r



    def load_all_data_tm(self,tm, calib=None, mapping=None):
        '''Loads all rows of data for a particular target moduel into memory

            Args:
                tm (int):   The slot number of the target module

            Kwargs:
                calib (arraylike): an array with calibration coefficient that should be applied to the data
                mapping (str or arraylike): a string to select a mapping  or an array with the mapping
                                            ['ssl2colrow','ssl2asic_ch','raw']
        '''
        if calib is None:
            calib = 1.0
        if(mapping is None):
            mapping = self.map
        elif(isinstance(mapping,str)):
            if(mapping == 'raw'):
                mapping = np.arange(N_TM_PIX)
            else:
                try:
                    mapping = ss_mappings.__getattribute__(mapping)
                except:
                    raise ValueError('No mapping found with name %s'%mapping)


        amps = np.zeros((self.n_readouts,N_TM_PIX))
        time = np.zeros(self.n_readouts)
        iro = np.zeros(self.n_readouts,dtype=np.uint64)

        for i, r in enumerate(self.read()):
            amps[i,:] = self.raw_readout[tm,:]*calib
            time[i] = self.readout_time
            iro[i] = self.readout_number



        amps = amps[:,mapping]

        ssdata = _nt('ssdata','iro amps time tm')
        return ssdata(iro,amps,time,tm)

    def __str__(self):
        s = 'SSDataReader:\n'
        s+='    Filename:%s\n'%self.filename
        s+='    Title: CHEC-S Slow signal monitor data\n'
        s+='    n_readouts: %d\n'%self.n_readouts
        s+='    ssdata-verion: %d\n'%self.attrs.ssdata_version
        s+='    data_type: %s\n'%self.data_type
        if(self.data_type=='Simulation'):
             s+='    number of simulated sources: %s\n'%len(self.sim_tables)

        return s