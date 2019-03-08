from ssm.core.pchain import ProcessingModule
import numpy as np

class FlatNSB(ProcessingModule):
    def __init__(self,rate):
        super().__init__('FlatNSB')
        self.rate = rate
    def configure(self,config):
        pass

    def run(self,frame):
            frame['raw_response'] += np.random.normal(loc=self.rate,
                                                  scale=0.05*self.rate,
                                                  size = frame['raw_response'].shape)
            return frame