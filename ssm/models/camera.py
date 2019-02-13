import numpy as np
import target_calib
_c = target_calib.CameraConfiguration("1.1.0")
_mapping = _c.GetMapping()
xv = np.array(mapping.GetXPixVector())*1e3
yv = np.array(mapping.GetYPixVector())*1e3

class SSDetectorResponseModel:
    def __init__(self,pix_resp, pix_posx=xv,pix_posy=yv):#,pix_size,prate2mV):
        self.pix_resp = pix_resp
        self.pix_posx = pix_posx
        self.pix_posy = pix_posy
        self.pix_pos = np.array(list(zip(pix_posx,pix_posy)))
#         self.pix_size = pix_size
#         self.prate2mV = prate2mV

    def response(self,path):
        path = np.asarray(path)
        res = np.zeros((len(path),len(self.pix_pos)))
        for i, p in enumerate(path):
            for j, pp in enumerate(self.pix_pos):
                if np.linalg.norm(p-pp) > 6:#self.pix_resp.m_size : #FIXME: corners not considered yet
#                     res[i,j] = 0
                    continue
                res[i,j] = self.pix_resp(p[0]-pp[0],p[1]-pp[1])
        return res