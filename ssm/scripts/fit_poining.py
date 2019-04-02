from importlib import reload
import tables
import numpy as np
import dashi

dashi.visual()
import yaml

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from CHECLabPy.plotting.camera import CameraImage, CameraImageImshow
from CHECLabPy.utils.mapping import (
    get_clp_mapping_from_tc_mapping,
    get_superpixel_mapping,
    get_tm_mapping,
)
import ssdaq
from ssm.core.pchain import ProcessingChain, ProcessingModule

from ssm.core.sim_io import DataReader

reader = DataReader("VegaTrack100minFlatNSB.hdf")
print(reader)

print("Simulation config:", reader.sim_attr_dict)


res = []
res_r = []
s_pos = []
times = []
import copy

for r in reader.read():
    res.append(r.flatten())
    res_r.append(reader.raw_data.flatten())
    times.append(reader.cpu_t)
    s_pos.append(reader.source_pos)


from tqdm.auto import tqdm


def compute_neigbor_map(pixdist=1.6):
    from target_calib import CameraConfiguration

    c = CameraConfiguration("1.1.0")
    m = c.GetMapping()

    xpix = np.array(m.GetXPixVector())
    ypix = np.array(m.GetYPixVector())
    size = m.GetSize()
    dist = pixdist * size
    neighbors = []  # [[]]*2048
    for i in range(2048):
        neighbors.append([])
    for i in tqdm(range(2048), total=2048, desc="Mapping neighbor pixels"):
        for j in range(i + 1, 2048):

            dx = xpix[i] - xpix[j]
            dy = ypix[i] - ypix[j]

            if np.sqrt(dx ** 2 + dy ** 2) < (dist):
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


neighbors = compute_neigbor_map()


def cluster_build(ind, data, lot, visited, neighbors):
    cluster = []
    visited.append(ind)
    if data[ind] > lot:
        cluster.append(ind)
        for n in neighbors[ind]:
            if n in visited:
                continue
            cluster += cluster_build(n, data, lot, visited, neighbors)
    return cluster


def find_clusters(res, upthreshold, lothreshold, neighbors):
    peaks = np.where(res > upthreshold)
    clusters = []
    for peak in peaks[0]:
        for c in clusters:
            if peak in c:
                break
        else:
            clusters.append(cluster_build(peak, res, lothreshold, [peak], neighbors))

    return clusters


from collections import defaultdict


def get_cluster_evolutionr2(clusters, data, neighbors, upthreshold, lothreshold):
    clusters_data = []
    pix_clust_evs = []
    for p in tqdm(
        clusters, total=len(clusters), desc="determining cluster time evolution"
    ):
        peak = p[0]
        pix_ind = set(p)
        cluster_data = defaultdict(list)
        pix_clust_ev = []
        for t, r in enumerate(data):
            if r[peak] < lothreshold:
                break
            ci = cluster_build(peak, r, lothreshold, [peak], neighbors)
            past_ci = []
            if len(pix_clust_ev) > 0:
                for pci in pix_clust_ev[-1]:
                    if r[pci] > lothreshold:
                        past_ci.append(pci)
                ci = list(set(past_ci).union(ci))
            pix_ind = pix_ind.union(ci)
            peak = ci[np.argmax(r[ci])]
            for k, v in zip(ci, r[ci]):
                cluster_data[k].append((t, v))
            pix_clust_ev.append(ci)
        clusters_data.append(cluster_data)
        pix_clust_evs.append(pix_clust_ev)
    return clusters_data, pix_clust_evs


def clean_ss_data(data, neighbors, upthreshold, lothreshold):
    clusters = find_clusters(res[0], upthreshold, lothreshold, neighbors)
    sress, ipixs = get_cluster_evolutionr2(
        clusters, res, neighbors, upthreshold, lothreshold
    )
    return sress, ipixs


sress, ipixs = clean_ss_data(res, neighbors, 170, 90)


from ssm.star_cat.hipparcos import load_hipparcos_cat

dt, stars = load_hipparcos_cat()
from astropy.coordinates import SkyCoord

vega = SkyCoord.from_name("vega")


from ssm.fit import fit, fitmodules

pfit = fit.PointingFit("response", "data")


from ssm.models import pixel_model

pixm = pixel_model.PixelResponseModel.from_file("ssm/resources/testpix_m.hkl")
feeder = fitmodules.DataFeeder(sress[7], times)
tel = fitmodules.TelescopeModel()
prstr = fitmodules.ProjectStarsModule(dt, stars)
illu = fitmodules.IlluminationModel(pixm)
nsb = fitmodules.FlatNSB()
from ssm.models.calibration import RateCalibration

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

# pfit.params[1].val = vega.dec.deg+0.000277777778*14
print(pfit.fit_chain)
print(pfit.params)
from scipy import optimize

method = "BFGS"  #'Nelder-Mead'#
rs = optimize.minimize(
    pfit, [v.val for v in pfit.params], method=method, tol=1.0, options={"disp": True}
)

print(rs)
print(np.diag(rs.hess_inv))
