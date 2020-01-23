from ssm.core.pchain import ProcessingModule
import numpy as np
from ssm.star_cat import hipparcos


class StarLightPropagator(ProcessingModule):
    def __init__(self, pix_model):
        super().__init__("StarLightPropagator")
        self.pix_model = pix_model

    def configure(self, config):
        self.lc = hipparcos.LightConverter()
        self.pix_pos = config["pix_pos"]

    def run(self, frame):
        res = np.zeros(2048)
        star_srcs = frame["star_sources"]
        for spos, rate in zip(
            star_srcs.pos, self.lc.mag2photonrate(star_srcs.vmag, area=6.5) * 1e-6
        ):
            self._raw_response(spos, res, rate)
        frame["raw_response"] = res
        return frame

    def _raw_response(self, source, res, rate):
        l = np.linalg.norm(source - self.pix_pos, axis=1)
        i = np.where(l < self.pix_model.model_size)[0]
        x = source[0] - self.pix_pos[i][:, 0]
        y = source[1] - self.pix_pos[i][:, 1]
        res[i] += self.pix_model(x, y, grid=False) * rate
