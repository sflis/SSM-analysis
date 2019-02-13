import numpy as np
from scipy import interpolate

class PSFModel:
    def __init__(self,radialsize,params):
        self.radialsize
        self.params = params

    def __call__(self,x,y):
        raise NotImplementedError

    def get_parameters(self):
        return self.params

    def set_parameters(self,params):
        self.params = params

    parameters = property(get_parameters, set_parameters)

class RadialPSFModel:
    def __init__(self,rho,y,norm):
        self.rho = rho
        self.y = y
        self.k = 3
        self.norm = norm
        self.spl = interpolate.CubicSpline(self.rho,self.y,bc_type=((1, 0.0), (1, 0.0)),extrapolate=True )#interpolate.UnivariateSpline(self.rho,self.y,ext=3,k=self.k)#,s=1e20)
        # self.params = np.array([norm]+list(y))
        # self.radialsize = np.max(rho)
        super().__init__(np.max(rho),np.array([norm]+list(y)))

    def __call__(self,x,y):
        rho,phi = cart2pol(x,y)
        return self.eval_polar(rho,phi)

    def eval_polar(self,rho,phi):
        rho = np.asarray(rho)
        y = np.asarray(self.norm*self.spl(rho))
        y[(y<0) | (rho>self.rho[-1])] = 0
        return y

    # def get_parameters(self):
    #     return self.params

    def set_parameters(self,params):
        self.params = params
        self.y = params[1:]
        self.norm = params[0]
        self.spl =interpolate.CubicSpline(self.rho,self.y,bc_type=((1, 0.0), (1, 0.0)) )

