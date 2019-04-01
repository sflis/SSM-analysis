from ssm.core.pchain import ProcessingModule


class Print(ProcessingModule):
    def __init__(self, name, print_keys=None, sel=1):
        super().__init__(name)
        self.print_keys = print_keys
        self.sel = sel
        self.n_frames = 0

    def configure(self, config):
        self.n_frames = 0

    def run(self, frame):
        if self.n_frames % self.sel == 0:
            print("+++++ %s +++++" % self.name)
            for k, v in frame.items():
                print("{}: {}".format(k, v))
        self.n_frames += 1
        return frame


class Aggregate(ProcessingModule):
    def __init__(self, keys, name="Aggregate"):
        super().__init__(name)
        self.keys = keys
        self.aggr = {}

        for k in self.keys:
            self.input[k] = k
            self.aggr[k] = []

    def clear(self):
        for k, v in self.aggr.items():
            self.aggr[k] = []

    def configure(self, config):
        pass

    def run(self, frame):
        for k in self.keys:
            self.aggr[k].append(frame[k])
        return frame
