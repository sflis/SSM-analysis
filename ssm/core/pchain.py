from tqdm.auto import tqdm
import numpy as np

class ProcessingChain:
    def __init__(self):
        self.chain = []
        self.config = {}
        self.frame_n = 0
        self.config_run = False
        self.num_funcs = 0
    def add(self, module):
        if callable(module):
            self.chain.append(FuncModule(module,'func%d'%self.num_funcs))
            self.num_funcs +=1
        else:
            self.chain.append(module)
        # else:
        #     raise ValueError('Function or class not compatible as ProcessingModule')

    def __str__(self):
        s = "This processing chain contains {} modules:\n".format(len(self.chain))
        for m in self.chain:
            s += "  {}\n".format(m.__str__())
        return s

    def configure(self):
        self.config = {'modules':[m.name for m in self.chain]}
        self.frame_n = 0
        for module in self.chain:
            # try:
            module.configure(self.config)
            # except Exception as e:
        self.config_run = True

    def mod_list(self):
        for mod in self.chain:
            print(mod.name)

    def _next(self):
        frame = {}
        while frame is not None:
            self.frame_n += 1
            frame = {'frame_n':self.frame_n}
            frame = self.chain[0].run(frame)
            if frame is None or self.stop:
                break
            yield frame

    def run(self,max_frames = None):
        if not self.config_run:
            self.configure()
        self.config_run = False
        n_frames = max_frames
        if 'n_frames' in self.config :
            n_frames = self.config['n_frames'] if max_frames is None else max_frames
        kwargs = {'total':n_frames, 'mininterval':0}
        self.stop = False
        for frame in tqdm(self._next(),**kwargs):
            for module in self.chain[1:]:
                frame =module.run(frame)
            if(n_frames is not None and self.frame_n>=n_frames):
                self.stop = True
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

    def __str__(self):
        return "<{}>:{}".format(self.__class__.__name__,self._name)


class FuncModule(ProcessingModule):
    def __init__(self,func,name):
        super().__init__(name)
        self.func = func

    def configure(self,config):
        pass

    def run(self,frame):
        return self.func(frame)

