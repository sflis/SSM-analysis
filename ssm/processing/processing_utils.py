import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
from numpy import ndarray

def compute_pixneighbor_map(config, pixdist: float = 1.6):
    m = config.GetMapping()

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


def cluster_build(ind: int, data: ndarray, lot: float, visited: list, neighbors: list):
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
):
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


def clean_ss_data(data, neighbors, upthreshold, lothreshold):
    clusters = find_clusters(res[0], upthreshold, lothreshold, neighbors)
    sress, ipixs = get_cluster_evolution(
        clusters, res, neighbors, upthreshold, lothreshold
    )
    return sress, ipixs
