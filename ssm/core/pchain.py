from tqdm.auto import tqdm
import numpy as np

class ProcessingChain:
    def __init__(self):
        self.chain = []
        self.config = {}
        self.frame_n = 0
        self.config_run = False
    def add(self, module):
        self.chain.append(module)

    def __str__(self):
        s = ''
        for m in self.chain:
            s += m.__str__()+'\n'
        return s
    def configure(self):
        self.config = {}
        self.frame_n = 0
        for module in self.chain:
            # try:
            module.configure(self.config)
            # except Exception as e:
        self.config_run = True
    def mod_list(self):
        for mod in self.chain:
            print(mod.name)

    def run(self,max_frames = None):
        if(not self.config_run):
            self.configure()
        self.config_run = False
        n_frames = self.config['n_frames'] if max_frames is None else max_frames
        for i in tqdm(range(n_frames)):
            self.frame_n += 1
            frame = {'frame_n':self.frame_n}
            for module in self.chain:
                frame =module.run(frame)

        for module in self.chain:
            frame = module.finish(self.config)

class ProcessingModule:
    def __init__(self,name):
        self._name = name

    def configure(self,config):
        raise NotImplementedError

    def run(self,frame):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    def finish(self,config):
        pass