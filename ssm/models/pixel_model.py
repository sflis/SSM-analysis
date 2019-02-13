import numpy as np


def create_pixel_response_model(psf,pixel_size,nbins):
    from tqdm import tqdm
    from collections import namedtuple
    model_size = pixel_size/2 + psf.radialsize
    xi, yi = np.meshgrid(np.linspace(-model_size,model_size,nbins),
                         np.linspace(-model_size,model_size,nbins))
    d = np.diff(xi.flatten())[0]
    path = list(zip(xi.flatten(),yi.flatten()))
    response = np.empty(len(path))
    for i, p in tqdm(enumerate(path),total=len(path)):
        response[i] = integ_psf(psf,p,pixel_size,epsabs=0.1)[0]
    response /= np.max(response)
    Response = namedtuple('PixelPSFResponse','response xi yi m_size')
    Response.__doc__ = """The Pixel-PSF normalized response
                          :param ndarray response: the response
                          :param ndarray xi: x coordinates in the meshgrid of the response
                          :param ndarray yi: y coordinates in the meshgrid of the response"""
    return Response(response.reshape(xi.shape),xi,yi,model_size)

def get_inter_pix_res(res_model):
    import scipy
    return scipy.interpolate.RectBivariateSpline(res_model.yi[:,0],res_model.xi[0,:],res_model.response)