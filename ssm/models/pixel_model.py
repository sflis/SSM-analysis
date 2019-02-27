import numpy as np
from collections import namedtuple
from ssm.utils.model_tools import integ_psf_on_pixel
import hickle as hkl
import pickle
class PixelResponseModel:
    def __init__(self,response,xi,yi,pixel_size):
        from scipy.interpolate import RectBivariateSpline
        if(np.max(xi)!=-np.min(xi) or np.max(yi)!=-np.min(yi)):
            raise ValueError('The response needs to be defined in a square')

        self.response = response
        self.xi = xi
        self.yi = yi
        self.pixel_size = pixel_size
        self.response_spl = RectBivariateSpline(yi[:,0],
                                                xi[0,:],
                                                response
                                                )

        self.model_size = np.max(xi)
        self.psf = None
    @classmethod
    def from_file(cls,filename):
        loaded = pickle.load(open(filename,'rb'))
        pix_m = loaded['pix_m']
        cons = cls(pix_m.response,pix_m.xi,pix_m.yi,pix_m.pixel_size)
        cons.psf = loaded['psf']
        return cons

    @classmethod
    def from_psf(cls, psf,pixel_size,nbins=40):
        from tqdm.auto import tqdm
        model_size = pixel_size/2 + psf.radialsize
        xi, yi = np.meshgrid(np.linspace(-model_size,model_size,nbins),
                             np.linspace(-model_size,model_size,nbins))
        d = np.diff(xi.flatten())[0]
        path = list(zip(xi.flatten(),yi.flatten()))
        response = np.empty(len(path))
        for i, p in tqdm(enumerate(path),total=len(path)):
            response[i] = integ_psf_on_pixel(psf,p,pixel_size,epsabs=0.1)[0]
        response /= np.max(response)
        cons = cls(response.reshape(xi.shape),xi,yi,pixel_size)
        cons.psf = psf
        return cons

    def save_model(self, filename):
        pickle.dump({'psf':self.psf,'pix_m':self},open(filename,'wb'))


    def __call__(self,x,y,grid=False):
        return self.response_spl(x,y,grid=grid)


PixelResponse = namedtuple('PixelPSFResponse','response xi yi m_size pixel_size')
PixelResponse.__doc__ = """The Pixel-PSF normalized response
                          :param ndarray response: the response
                          :param ndarray xi: x coordinates in the meshgrid of the response
                          :param ndarray yi: y coordinates in the meshgrid of the response"""

def create_pixel_response_model(psf,pixel_size,nbins):
    # from tqdm import tqdm
    from tqdm.auto import tqdm
    model_size = pixel_size/2 + psf.radialsize
    xi, yi = np.meshgrid(np.linspace(-model_size,model_size,nbins),
                         np.linspace(-model_size,model_size,nbins))
    d = np.diff(xi.flatten())[0]
    path = list(zip(xi.flatten(),yi.flatten()))
    response = np.empty(len(path))
    for i, p in tqdm(enumerate(path),total=len(path)):
        response[i] = integ_psf_on_pixel(psf,p,pixel_size,epsabs=0.1)[0]
    response /= np.max(response)

    return PixelResponse(response.reshape(xi.shape),xi,yi,model_size,pixel_size)

def get_inter_pix_res(res_model):
    import scipy
    SplinedPixelModel = namedtuple('SplinedPixelModel','px_md_spl px_md')

    return SplinedPixelModel(scipy.interpolate.RectBivariateSpline(res_model.yi[:,0],
                                                                    res_model.xi[0,:],
                                                                    res_model.response
                                                                    ),
                            res_model)