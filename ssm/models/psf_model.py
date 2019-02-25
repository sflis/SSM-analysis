import numpy as np
from scipy import interpolate
from ssm.utils.model_tools import cart2pol, pol2cart

class PSFModel:
    def __init__(self,radialsize,params):
        self.radialsize = radialsize
        self.params = params

    def __call__(self,x,y):
        raise NotImplementedError

    def get_parameters(self):
        return self.params

    def set_parameters(self,params):
        self.params = params

    parameters = property(get_parameters, set_parameters)


class RadialPSFModel(PSFModel):
    def __init__(self,rho,y,norm):
        self.rho = rho
        self.y = y
        self.k = 3
        self.norm = norm
        self.spl = interpolate.CubicSpline(self.rho,self.y,bc_type=((1, 0.0), (1, 0.0)),extrapolate=True )#interpolate.UnivariateSpline(self.rho,self.y,ext=3,k=self.k)#,s=1e20)
        super().__init__(np.max(rho),np.array([norm]+list(y)))

    def __call__(self,x,y):
        rho,phi = cart2pol(x,y)
        return self.eval_polar(rho,phi)

    def eval_polar(self,rho,phi):
        rho = np.asarray(rho)
        y = np.asarray(self.norm*self.spl(rho))
        y[(y<0) | (rho>self.rho[-1])] = 0
        return y


    def set_parameters(self,params):
        self.params = params
        self.y = params[1:]
        self.norm = params[0]
        self.spl =interpolate.CubicSpline(self.rho,self.y,bc_type=((1, 0.0), (1, 0.0)) )

def gaussian2d(x,y,x0,y0,a,b,c,norm):
  return norm*np.exp(-a*(x - x0)**2 +2*b*(x-x0)*(y-y0)-c*(y - y0)**2)

class BinormalPSF(PSFModel):
    def __init__(self,x0,y0,sigx,sigy,sigxy,norm,rsize):
        self.x0 = x0
        self.sigx = sigx
        self.y0 = y0
        self.sigy = sigy
        self.rsize = rsize
        self.sigxy = sigxy
        self.norm = norm
        super().__init__(rsize,np.array([x0,y0,sigx,sigy,sigxy,norm,rsize]))


    def __call__(self,x,y):
        return gaussian2d(x,y,self.x0,self.y0,self.sigx,self.sigxy,self.sigy,self.norm)

    def eval_polar(self,rho,phi):
        x,y = pol2cart(rho,phi)
        return self.__call__()


    def set_parameters(self,params):
        self.params = params
        self.x0 = params[0]
        self.y0 = params[1]
        self.sigx = params[2]
        self.sigy = params[3]
        self.sigxy = params[4]
        self.norm = params[5]
        self.rsize = params[6]

