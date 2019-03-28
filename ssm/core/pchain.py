from tqdm.auto import tqdm
import numpy as np
import inspect


class ProcessingChain:
    def __init__(self):
        self.chain = []
        self.config = {}
        self.frame_n = 0
        self.config_run = False
        self.num_funcs = 0

    def add(self, module):

        if callable(module):
            self.chain.append(
                FuncModule(module, "func%d ('%s')" % (self.num_funcs, module.__name__))
            )
            self.num_funcs += 1
        elif issubclass(type(module), ProcessingModule):
            self.chain.append(module)
        else:
            raise ValueError(
                "Class {} not compatible as ProcessingModule".format(module.__class__)
            )

    def __str__(self):
        s = "This processing chain contains {} modules:\n".format(len(self.chain))
        for m in self.chain:
            # ms = m.__str__()
            # ms = ms.split("\n")
            # mlen = max([len(s) for s in ms])
            # nms = ["-"*(mlen+2)+"\n"]
            # for ss in ms:
            #     nms.append("|{}|\n".format(ss))
            # # nms = ["-"*(mlen+2)+"\n"]
            # s += "  {}\n".format("".join(nms))
            s += "  {}\n".format(m.__str__())
        return s

    def configure(self):
        self.config = {"modules": [m.name for m in self.chain]}
        self.frame_n = 0

        for module in self.chain:
            # try:
            module._introspection()
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
            frame = {"frame_n": self.frame_n}
            frame = self.chain[0].run(frame)
            if frame is None or self.stop:
                break
            yield frame

    def run(self, max_frames=None):
        # configuration stage
        if not self.config_run:
            self.configure()
        self.config_run = False
        n_frames = max_frames
        if "n_frames" in self.config:
            n_frames = self.config["n_frames"] if max_frames is None else max_frames
        kwargs = {"total": n_frames, "mininterval": 0}
        self.stop = False

        # running stage
        for frame in tqdm(self._next(), **kwargs):
            for module in self.chain[1:]:
                frame = module.run(frame)
                if frame is None:
                    break
            if n_frames is not None and self.frame_n >= n_frames:
                self.stop = True

        # finishing stage
        for module in self.chain:
            frame = module.finish(self.config)


class ProcessingModule:
    def __init__(self, name):
        self._name = name
        self._input = {}
        self._output = {}
        self._cinput = {}
        self._coutput = {}
        self._introspected = False

    def configure(self, config):
        raise NotImplementedError

    def run(self, frame):
        raise NotImplementedError

    def _introspection(self):
        from copy import copy

        if not self._introspected:
            # Introspecting to find all input parameters that should be
            # changed to properties
            for iok, iov in self.__dict__.items():
                if iok[:4] == "out_":
                    self._output[iok[4:]] = iov
                    setattr(
                        self.__class__,
                        iok,
                        property(
                            lambda self, k=iok[4:]: self._output[k],
                            lambda self, v, k=iok[4:]: self._output.update({k: v}),
                        ),
                    )

                if iok[:3] == "in_":
                    self._input[iok[3:]] = iov
                    setattr(
                        self.__class__,
                        iok,
                        property(
                            lambda self, k=iok[3:]: self._input[k],
                            lambda self, v, k=iok[3:]: self._input.update({k: v}),
                        ),
                    )

                if iok[:5] == "cout_":
                    self._coutput[iok[5:]] = iov
                    setattr(
                        self.__class__,
                        iok,
                        property(
                            lambda self, k=iok[5:]: self._coutput[k],
                            lambda self, v, k=iok[5:]: self._coutput.update({k: v}),
                        ),
                    )
                if iok[:4] == "cin_":
                    self._input[iok[4:]] = iov
                    setattr(
                        self.__class__,
                        iok,
                        property(
                            lambda self, k=iok[4:]: self._cinput[k],
                            lambda self, v, k=iok[4:]: self._cinput.update({k: v}),
                        ),
                    )

            self._introspected = True

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def cinput(self):
        return self._cinput

    @property
    def coutput(self):
        return self._coutput

    @property
    def name(self):
        return self._name

    def finish(self, config):
        pass

    def __str__(self):
        self._introspection()
        s = ""
        for k, v in self._input.items():
            s += " {{{}}},".format(v)
        s = "↓" + s[1:-1] + "↓\n"

        s += "<{}>:{}: \n     →".format(self.__class__.__name__, self._name)
        for k, v in self._output.items():
            s += " {{{}}},".format(v)
        return s


class FuncModule(ProcessingModule):
    def __init__(self, func, name):
        super().__init__(name)
        self.func = func

    def configure(self, config):
        pass

    def run(self, frame):
        return self.func(frame)
