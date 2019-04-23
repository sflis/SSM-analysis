import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
from numpy import ndarray

def compute_pixneighbor_map(config, pixdist: float = 1.8,intertmdist: float = 2.5)->list:
    """ Computes a pixel neighbor map for given pixel distances
        within an TM and between pixels at adjecent TMs.
    """
    m = config.GetMapping()

    xpix = np.array(m.GetXPixVector())
    ypix = np.array(m.GetYPixVector())
    size = m.GetSize()
    dist = pixdist * size
    intdist = intertmdist*size
    neighbors = []  # [[]]*2048
    for i in range(2048):
        neighbors.append([])
    for i in tqdm(range(2048), total=2048, desc="Mapping neighbor pixels"):
        for j in range(i + 1, 2048):

            dx = xpix[i] - xpix[j]
            dy = ypix[i] - ypix[j]
            tm1 = int(i/64)
            tm2 = int(j/64)
            d = dist if tm1==tm2 else intdist
            if np.sqrt(dx ** 2 + dy ** 2) < (d):
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


def cluster_build(ind: int, data: ndarray, lot: float, visited: list, neighbors: list)->list:
    cluster = []
    visited.append(ind)
    if data[ind] > lot:
        cluster.append(ind)
        for n in neighbors[ind]:
            if n in visited:
                continue
            cluster += cluster_build(n, data, lot, visited, neighbors)
    return cluster


def find_clusters(
    res: ndarray, upthreshold: float, lothreshold: float, neighbors: list
)->list:
    peaks = np.where(res > upthreshold)
    clusters = []
    for peak in peaks[0]:
        for c in clusters:
            if peak in c:
                break
        else:
            clusters.append(cluster_build(peak, res, lothreshold, [peak], neighbors))

    return clusters


def get_cluster_evolution(
    clusters: list,
    data: ndarray,
    neighbors: list,
    upthreshold: float,
    lothreshold: float,
    verbose: bool = True,
):
    clusters_data = []
    pix_clust_evs = []
    for p in tqdm(
        clusters,
        total=len(clusters),
        desc="Determining cluster time evolution",
        disable=not verbose,
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

def evolve_clusters(
    data: list,
    neighbors: list,
    upthreshold: float,
    lothreshold: float,
    verbose: bool = True,
):
    clusters = []
    # last = set()
    it = 0
    for t,dat in enumerate(tqdm(
            data,
            total=len(data),
            desc="Determining cluster time evolution",
            disable=not verbose,
        )):
        new_clusters = find_clusters(dat, upthreshold, lothreshold, neighbors)
        merge = []
        added = 0

        for nc in new_clusters:
            addto = []
            for i, oc in enumerate(clusters):
                if 1+oc[-1][0]>=t and len(oc[-1][1].intersection(nc))>0:
                    addto.append(i)

            if addto:
                added +=1
                c = clusters[addto[0]]
                for pix in nc:
                    c[pix].append((t,dat[pix]))
                c[-2] = (t,c[-2][1].union(nc))
                if len(addto)>1:
                    merge.append(addto)
            else:

                cluster_data = defaultdict(list)
                for pix in nc:
                    cluster_data[pix].append((t,dat[pix]))
                cluster_data[-1] = (t,set(nc))
                cluster_data[-2] = (t,set(nc))

                clusters.append(cluster_data)
        for c in clusters:

            if(c[-2][0] < t):
                index = np.array(list(c[-1][1]),dtype=np.uint64)
                sub_index = np.argwhere(dat[index]>lothreshold)
                if len(sub_index)>0:
                    nc = set(index[sub_index[0]])
                    c[-2] = (t,nc)
                    for pix in nc:
                        c[pix].append((t,dat[pix]))
            c[-1] = c[-2]
        to_del = []
        for m in merge:
            c_main = clusters[m[0]]
            for mi in reversed(m[1:]):
                to_del.append(mi)
                c = clusters[mi]
                c_main[-1] = (t,c_main[-1][1].union(c[-1][1]))
                for pix,ts in c.items():
                    c_main[pix] +=ts

        for mi in reversed(sorted(list(set(to_del)))):
            del clusters[mi]
    for c in clusters:
        del c[-1]
        del c[-2]
    return clusters


def cluster_cleaning(data, neighbors, upthreshold, lothreshold):
    clusters = find_clusters(res[0], upthreshold, lothreshold, neighbors)
    sress, ipixs = get_cluster_evolution(
        clusters, res, neighbors, upthreshold, lothreshold
    )
    return sress, ipixs

def smooth_slowsignal(a, n=10):
    """ Simple smoothing algorithm that uses a moving average between readout frames
        It assumes that the time between each readout is equidistant.
    """
    ret = np.cumsum(a, axis = 0,dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n