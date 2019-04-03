import numpy as np

from ssm.core.pchain import ProcessingChain, ProcessingModule
from ssm.core.util_pmodules import Aggregate
from ssm.processing.processingmodules import Reader,ClusterCleaning

data_proc = ProcessingChain()
reader = Reader('VegaTrack100minFlatNSB.hdf')
reader.out_raw_resp = "raw_resp"
reader.out_time = "time"
c_cleaner = ClusterCleaning(170.0,90.0)
c_cleaner.in_data = reader.out_raw_resp
c_cleaner.out_cleaned = 'cluster_cleaned'
aggr = Aggregate([reader.out_raw_resp,c_cleaner.out_cleaned,reader.out_time])
data_proc.add(reader)
data_proc.add(c_cleaner)
data_proc.add(aggr)
print(data_proc)
data_proc.run()


from ssm.star_cat.hipparcos import load_hipparcos_cat
from ssm.models import pixel_model
from ssm.models.calibration import RateCalibration
from ssm.fit import fit, fitmodules
from astropy.coordinates import SkyCoord

dt, stars = load_hipparcos_cat()


vega = SkyCoord.from_name("vega")
pfit = fit.PointingFit("response", "data")
pixm = pixel_model.PixelResponseModel.from_file("ssm/resources/testpix_m.hkl")
feeder = fitmodules.DataFeeder(aggr.aggr[c_cleaner.out_cleaned][0][7], aggr.aggr[reader.out_time][0])
tel = fitmodules.TelescopeModel()
prstr = fitmodules.ProjectStarsModule(dt, stars)
illu = fitmodules.IlluminationModel(pixm)
nsb = fitmodules.FlatNSB()


cal = RateCalibration("SteveCalSmoothCutoff")
resp = fitmodules.Response(cal)

tel.par_pointingra.val = vega.ra.deg - 0.000277777778 * 12
tel.par_pointingdec.val = vega.dec.deg + 0.000277777778 * 14
pfit.addFitModule(feeder)
pfit.addFitModule(tel)
pfit.addFitModule(prstr)
pfit.addFitModule(illu)
pfit.addFitModule(nsb)
pfit.addFitModule(resp)

print(pfit.fit_chain)
print(pfit.params)
from scipy import optimize

method = "BFGS"  #'Nelder-Mead'#
rs = optimize.minimize(
    pfit, [v.val for v in pfit.params], method=method, tol=1.0, options={"disp": True}
)

print(rs)
print(np.diag(rs.hess_inv))
