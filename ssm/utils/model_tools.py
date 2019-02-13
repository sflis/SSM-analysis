import numpy as np
from scipy import integrate


class Line:
    def __init__(self,p1,p2):
        p1,p2 = np.asarray(p1),np.asarray(p2)
        self.p = p1
        self.dir = p2-p1
        self.dir /=np.linalg.norm(self.dir)
    def __call__(self,l):
        ones = np.ones(l.shape)
        return np.outer(ones,self.p)+np.outer(ones,self.dir)*np.outer(np.ones(2),l).T


def integ_psf_on_pixel(psf,p,size,epsabs=0.1):
    hsize =0.5*size
    xa,xb = p[0]-hsize,p[0]+hsize
    ya,yb = p[1]-hsize,p[1]+hsize
    return integrate.dblquad(psf,xa,xb,lambda x:ya,lambda x:yb,epsabs=epsabs)



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)