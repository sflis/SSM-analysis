from ssm.core.pchain import ProcessingModule, ProcessingChain
from ssm.core.util_pmodules import Aggregate
import numpy as np


class PointingFit:
    def __init__(self, expectation_key, data_key):
        self.fit_chain = ProcessingChain(silent=True)
        # self.fit_chain.add(self)
        self.configured = False
        self.params = []
        self.expectation_key = expectation_key
        self.data_key = data_key

    def addFitModule(self, module):
        self.params += module.parameters.values()
        self.fit_chain.add(module)

    def _configure(self):
        if not self.configured:
            self.aggregator = Aggregate([self.expectation_key, self.data_key])
            self.fit_chain.add(self.aggregator)
            self.configured = True

    def compute_exp(self, x=None):
        if x is not None:
            for i, xp in enumerate(x):
                self.params[i].val = xp
        if not self.configured:
            self._configure()
        self.fit_chain.run()
        self.fit_chain.config_run = True
        exp = self.aggregator.aggr[self.expectation_key][0]
        self.data = self.aggregator.aggr[self.data_key][0]
        self.aggregator.clear()
        return exp

    def __call__(self, x):
        print(x)
        exp = self.compute_exp(x)
        # m = self.data>0
        chi2 = 0
        for expv, datav in zip(exp.items(), self.data.items()):
            chi2 += np.sum((expv[1][1] - datav[1][1]) ** 2 / datav[1][1])
        # chi2 = np.sum((exp - self.data) ** 2/self.data)
        print(chi2)
        return chi2


class FitModel(ProcessingModule):
    def __init__(self, name):
        super().__init__(name)
        self._par = {}
        self._par_registered = False

    def _registerparams(self):
        if not self._par_registered:
            # Introspecting to find all input parameters that should be
            # changed to properties
            for iok, iov in self.__dict__.items():
                if iok[:4] == "par_":
                    self._par[iok[4:]] = iov
                    setattr(
                        self.__class__,
                        iok,
                        property(
                            lambda self, k=iok[4:]: self._par[k],
                            lambda self, v, k=iok[4:]: self._par.update({k: v}),
                        ),
                    )
        self._par_registered = True

    @property
    def parameters(self):
        if not self._par_registered:
            self._registerparams()
        return self._par


class FitParameter:
    def __init__(self, name, val0, interval=None):
        self.name = name
        self.val = val0
        self.interval = interval

    def __repr__(self):
        return "{}: {} [{},{}]".format(self.name, self.val, *self.interval)
