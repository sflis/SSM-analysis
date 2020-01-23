import matplotlib.pyplot as plt
import numpy as np
from ssm.star_cat.hipparcos import load_hipparcos_cat
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from ssm.core import pchain
from ssm.pmodules import *
from ssm.core.util_pmodules import Aggregate, SimpleInjector
from ssm import pmodules
import copy
from CHECLabPy.plotting.camera import CameraImage
import os
from datetime import datetime
import dashi
from collections import defaultdict
import pickle
from scipy.optimize import minimize

dashi.visual()


def gaussian(x, y, x0, y0, xalpha, yalpha, A):
    return A * np.exp(-(((x - x0) / xalpha) ** 2) - ((y - y0) / yalpha) ** 2)


def fit_gauss(xy, z, guess):
    def f(params):
        return np.sum(np.abs(z - gaussian(xy[:, 0], xy[:, 1], *params)))

    res = minimize(f, guess)
    return res.x[0], res.x[1]


def get_fc_hotspots(data, clusters, frame_n):
    focal_pos = []
    brightness = []
    for c in clusters[frame_n]:
        # finding the brightest pixel as a rough estimate of center
        # then remove any pixels that are more than 2 cm from the brightest pixel
        # to supress ghosts
        pos = data.pix_pos[c]
        max_i = np.argmax(data.data[frame_n, c])
        d = np.linalg.norm(pos - pos[max_i], axis=1)
        m = d < .023

        focal_pos.append(np.average(pos[m], weights=data.data[frame_n, c][m], axis=0))
        brightness.append(np.sum(data.data[frame_n, c]))
    focal_pos = np.array(focal_pos)
    focal_pos = focal_pos[np.argsort(brightness)[::-1]]
    return focal_pos

def main(inputfile, image_path):
    sdt, stars = load_hipparcos_cat()
    # These are the stars we use to identify the patch of sky in
    # the FOV
    cvmag_lim = 6.6
    # catalog_star_table = sdt[sdt.vmag < cvmag_lim]
    # catalog_stars = stars[sdt.vmag < cvmag_lim]
    # Define telescope frame
    location = EarthLocation.from_geodetic(lon=14.974609, lat=37.693267, height=1750)
    obstime = Time("2019-05-09T01:37:54.728026")
    altaz_frame = AltAz(location=location, obstime=obstime)

    # Get pixel coordinates from TargetCalib
    from target_calib import CameraConfiguration

    camera_config = CameraConfiguration("1.1.0")
    mapping = camera_config.GetMapping()
    focal_length = u.Quantity(2.15191, u.m)
    from ssm.pointing.astrometry import (
        # rotang,
        # rot_matrix,
        # generate_hotspots,
        StarPatternMatch,
        # matchpattern,
    )

    matcher = StarPatternMatch.from_location(
        altaz_frame=altaz_frame,
        stars=stars,
        sdt=sdt,
        fov=12,
        focal_length=focal_length,
        min_alt=-90,
        vmag_lim=cvmag_lim,
        pixsize=mapping.GetSize(),
    )

    # initializing our process chain
    data_proc = pchain.ProcessingChain()

    reader = Reader(inputfile)
    data_proc.add(reader)

    # This module removes incomplete frames and marks bad and unstable pixels
    frame_cleaner = PFCleaner()
    data_proc.add(frame_cleaner)

    cal = Calibrate()
    data_proc.add(cal)
    # # A simple flat field computation based on the first 7000 frames
    # sff = SimpleFF(0, 7000, star_thresh=140)
    # sff.in_data = "calibrated_data"
    # sff.out_ff = "simple_ff_calibrated"
    # data_proc.add(sff)

    ff_compute = FlatFielding(16000, 26000, star_thresh=1.3)
    ff_compute.in_data = "calibrated_data"
    ff_compute.out_ff = "ff_calibrated"
    data_proc.add(ff_compute)

    # A simple flat field computation based on the first 7000 frames
    sff_on_raw = SimpleFF(0, 7000)
    data_proc.add(sff_on_raw)

    # The Aggregate module collects the computed object from the frame
    aggr = Aggregate(
        [
            "raw_resp",
            # "simple_ff",
            # "simple_ff_calibrated",
            "calibrated_data",
            "ff_calibrated",
        ]
    )
    data_proc.add(aggr)

    # Simple visualization of the chain
    print(data_proc)

    # Execute the chain
    data_proc.run()

    data = aggr.aggr["calibrated_data"][0]

    cffc = aggr.aggr["ff_calibrated"][0]

    proc_chain = pchain.ProcessingChain()
    # Copy data so that we do not overwrite the original data
    ffdata = copy.deepcopy(data)
    # apply the flatfielding
    ffdata.data -= cffc
    # The Simple injector just creates a frame with the content of the input dictionary
    injector = SimpleInjector({"data": ffdata})
    proc_chain.add(injector)
    # Smoothing the signal maybe not so useful right now
    smooth = SmoothSlowSignal(n_readouts=20)
    proc_chain.add(smooth)
    # Finds hotspot clusters
    clust = pmodules.ClusterCleaning(1.0, 0.9)
    clust.in_data = "data"  # smooth.out_data
    proc_chain.add(clust)

    # The Aggregate module collects the computed object from the frame
    # We want the clusters and the smooth data
    aggr = Aggregate(["clusters", "smooth_data"])
    proc_chain.add(aggr)
    proc_chain.run()

    # Extract the processed data
    clusters = aggr.aggr["clusters"][0]
    smooth_data = aggr.aggr["smooth_data"][0]

    plt.set_cmap("Greys_r")
    stop = 26000
    step = 100
    sep_list = []
    data_dict = defaultdict(list)
    for frame_n in tqdm(range(0, stop, step), total=stop / step):
        cluster_pos = get_fc_hotspots(smooth_data, clusters, frame_n)

        p = matcher.star_coordinates[matcher.star_table.hip_number.values == 85670]
        matched_hs = matcher.identify_stars(
            cluster_pos[:], horizon_level=25, search_region=(p, 17)
        )

        # Plotting
        fig, axs = plt.subplots(constrained_layout=True, figsize=(10 / 1.2, 6 / 1.2))
        # Different average camera images
        camera = CameraImage(
            smooth_data.xpix, smooth_data.ypix, smooth_data.pix_size, ax=axs
        )
        im = copy.deepcopy(smooth_data.data[frame_n])
        im[np.isnan(im)] = np.nanmean(im)
        camera.image = im
        print(im)
        draco = {85670: "beta", 87833: "gamma", 85829: "nu", 85819: "nu", 87585: "xi"}

        camera.add_colorbar("Rate (MHz)")

        camera.highlight_pixels(
            [item for sublist in clusters[frame_n] for item in sublist], color="r"
        )

        camera.set_limits_minmax(125, 200)
        axs.plot(cluster_pos[:, 0], cluster_pos[:, 1], "wo", mfc="none", ms=25, mew=2)
        axs.plot(cluster_pos[0, 0], cluster_pos[0, 1], "ro", mfc="none", ms=25, mew=4)
        bbox_props = dict(boxstyle="Round", fc="orange", ec="orange", lw=2, alpha=0.4)

        alt = 73.21 * u.deg
        az = 0.5 * u.deg
        obstime = Time(
            datetime.fromtimestamp(float(smooth_data.time[frame_n])), format="datetime"
        )
        axs.set_title(obstime)
        altaz_frame = AltAz(location=location, obstime=obstime)

        telescope_pointing = SkyCoord(alt=alt, az=az, frame=altaz_frame,)
        telsky = telescope_pointing.transform_to("icrs")
        data_dict["ra_true"].append(telsky.ra.rad)
        data_dict["dec_true"].append(telsky.dec.rad)
        data_dict["obstime"].append(obstime)
        data_dict["unixtime"].append(smooth_data.time[frame_n])

        if matched_hs is not None:

            for h in matched_hs:
                axs.plot(h[0][0], h[0][1], "go", mfc="none", ms=25, mew=2)
                print(h[1])
                if h[1] in draco:
                    axs.annotate(
                        draco[h[1]],
                        h[0],
                        h[0] + 0.01,
                        color="black",
                        size=14,
                        bbox=bbox_props,
                    )
                else:
                    axs.annotate('{}'.format(h[1]),
                                 h[0],
                                 h[0] - 0.01,
                                 size=9,
                                 color='red')

            try:
                ra, dec = matcher.determine_pointing(matched_hs)
                estimated_pointing = SkyCoord(ra=ra, dec=dec, unit="rad", frame="icrs")
                sep = estimated_pointing.separation(telescope_pointing)
                color = "green" if sep < 7 * u.arcsec else "red"
                sep_list.append(sep.arcsec)
                data_dict["status"].append(0)
                data_dict["ra_est"].append(ra)
                data_dict["dec_est"].append(dec)
                data_dict["sep"].append(sep.arcsec)
            except Exception:
                data_dict["status"].append(2)
                data_dict["ra_est"].append(np.nan)
                data_dict["dec_est"].append(np.nan)
                data_dict["sep"].append(np.nan)
                pass
        else:
            data_dict["ra_est"].append(np.nan)
            data_dict["dec_est"].append(np.nan)
            data_dict["status"].append(1)
            data_dict["sep"].append(np.nan)
        plt.savefig(os.path.join(image_path, "draco_im{:05d}".format(frame_n)))
        plt.close()
    hist_sep = dashi.histogram.hist1d(np.linspace(0, 300, 100))
    hist_sep.fill(np.array(sep_list))
    plt.figure()
    hist_sep.line()
    hist_sep.statbox()
    plt.title("Distribution of est-true pointing separations")
    plt.savefig(os.path.join(image_path, "separation_dist.png"))
    for k, v in data_dict.items():
        data_dict[k] = np.array(v)

    with open("draco_pointing_assessment_ghost_supr1.pkl", "wb") as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Start a simple Slow Signal readout listener.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p, --path",
        dest="path",
        type=str,
        default="AstrometryAssessment",
        help="Path to store images.",
    )
    parser.add_argument(
        "-f, --filename",
        dest="filename",
        type=str,
        default="/home/sflis/CTA/projects/SSM-analysis/data/astri_onsky/d2019-05-08/Run13312.hdf5",
        help="Filename",
    )

    args = parser.parse_args()
    main(inputfile=args.filename, image_path=args.path)
