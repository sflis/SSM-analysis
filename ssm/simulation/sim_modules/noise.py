from ssm.core.pchain import ProcessingModule
import numpy as np

class Noise(ProcessingModule):
    def __init__(self,resp_key = 'response', name='Noise'):
        super().__init__(name)
        self.resp_key = resp_key
    def configure(self,config):
        pass

    def run(self,frame):
        frame[self.resp_key] += np.random.normal(scale=0.01*frame[self.resp_key])
        return frame