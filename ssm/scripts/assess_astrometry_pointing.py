import numpy as np
import healpy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation

from ssdaq.data import Frame
from ssm.star_cat.hipparcos import load_hipparcos_cat
from target_calib import CameraConfiguration
from ssm.pointing.astrometry import generate_hotspots, StarPatternMatch

import zmq


def main(args):
    # tr = tracker.SummaryTracker()
    sdt, stars = load_hipparcos_cat()
    # These will be the stars we see in data
    source_star_table = sdt[sdt.vmag < 6.5]
    source_stars = stars[sdt.vmag < 6.5]

    # These are the stars we use to identify the patch of sky in
    # the FOV
    cvmag_lim = 6.1
    # Define telescope frame
    location = EarthLocation.from_geodetic(lon=14.974609, lat=37.693267, height=1750)
    obstime = Time("2019-05-09T01:37:54.728026")
    altaz_frame = AltAz(location=location, obstime=obstime)

    # Get pixel coordinates from TargetCalib

    camera_config = CameraConfiguration("1.1.0")
    mapping = camera_config.GetMapping()
    pos = np.array(
        [np.array(mapping.GetXPixVector()), np.array(mapping.GetYPixVector())]
    )
    focal_length = u.Quantity(2.15191, u.m)

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
    matcher.silence = True

    order = 10
    nside = healpy.order2nside(order)
    npix = healpy.nside2npix(nside)
    npixs_above_horizon = np.where(
        healpy.pix2ang(nside, np.arange(npix))[0] > np.pi / 2
    )[0]

    tstamp0 = 1557360406
    np.random.seed(args.seed)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    ip = "127.0.0.1"
    con_str = "tcp://%s:" % ip + str(args.port)
    socket.connect(con_str)

    obstime = Time(tstamp0 + np.random.uniform(-43200, 43200), format="unix")
    pixsize = mapping.GetSize()
    for i in range(args.n_iterations):
        print(f"Sample: {i}", flush=True)
        tmp_timestamp = tstamp0 + np.random.uniform(-43200, 43200)
        obstime = Time(tmp_timestamp, format="unix")
        altaz_frame = AltAz(location=location, obstime=obstime)
        pixid = int(np.random.uniform(0, npixs_above_horizon.shape[0]))
        ang = healpy.pix2ang(nside, pixid)
        alt = ang[0]
        az = ang[1]
        hotspots, tel_pointing, star_ind, hips_in_fov, all_hips = generate_hotspots(
            alt * u.rad,
            az * u.rad,
            altaz_frame,
            source_stars,
            source_star_table,
            pixsize,
            cvmag_lim,
            pos,
        )

        true_hotspots = np.array(hotspots)
        frame = Frame()
        frame.add("hips_in_fov", np.array(hips_in_fov))
        frame.add("hotspots", true_hotspots)
        tel_sky_pointing = tel_pointing.transform_to("icrs")
        frame.add(
            "tel_pointing",
            np.array([tel_sky_pointing.ra.rad, tel_sky_pointing.dec.rad]),
        )

        hotspots = true_hotspots.copy()
        N_change = 1
        # hotspots[N_change, :] = hotspots[N_change, :] + 0.003

        matched_hs = matcher.identify_stars(
            hotspots, horizon_level=0, obstime=obstime, only_one_it=False
        )


        if matched_hs is not None and len(matched_hs) > 0:
            ra, dec = matcher.determine_pointing(matched_hs)
            frame.add("matched_hs", np.array(matched_hs))
            frame.add("est_pointing", np.array([ra, dec]))
        else:
            print(tmp_timestamp, alt, az)

        matched_hs = matcher.identify_stars(
            hotspots, horizon_level=0, obstime=obstime, only_one_it=True
        )
        match = np.array(matched_hs)
        match_quantity = match[:, 2] * match[:, 3] * match[:, 1]
        index = np.where(match[:, 0] == all_hips[0][1])[0]
        matched_match = np.argmax(match_quantity)
        frame.add('true_match', match[index])
        frame.add('matched_match', match[matched_match])
        frame.add('match_quantity_list', match)

        socket.send(frame.serialize())


import argparse
from ssdaq.core.basesubscribers import BasicSubscriber, WriterSubscriber
from ssdaq.data import io


class FrameSubscriber(BasicSubscriber):
    def __init__(self, ip: str, port: int, logger=None):
        super().__init__(ip=ip, port=port, logger=logger, unpack=Frame.unpack)


class FrameFileWriter(WriterSubscriber):
    def __init__(
        self,
        file_prefix: str,
        ip: str,
        port: int,
        folder: str = "",
        file_enumerator: str = None,
        filesize_lim: int = None,
    ):

        super().__init__(
            subscriber=FrameSubscriber,
            writer=io.FrameWriter,
            file_ext=".icf",
            name="FrameFileWriter",
            **{k: v for k, v in locals().items() if k not in ["self", "__class__"]},
        )


def writer(args):

    data_writer = FrameFileWriter(
        args.filename, file_enumerator="order", port=args.port, ip="0.0.0.0"
    )

    data_writer.start()
    running = True
    while running:
        ans = input("To stop type `yes`: \n")
        if ans == "yes":
            running = False
    try:
        print("Waiting for writer to write buffered data to file......")
        print("`Ctrl-C` will empty the buffers and close the file immediately.")
        data_writer.close()
    except KeyboardInterrupt:
        print()
        data_writer.close(hard=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start a simple Slow Signal readout listener.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p, --port", dest="port", type=int, default=9999, help="port")
    parser.add_argument(
        "-n, --n-iterations",
        dest="n_iterations",
        type=int,
        default=100,
        help="Number of iterations",
    )
    parser.add_argument(
        "-w, --writer", dest="writer", action="store_true", help="start writer"
    )
    parser.add_argument(
        "-s, --seed",
        dest="seed",
        type=int,
        default=1,
        help="random number generator seed",
    )
    parser.add_argument(
        "-f, --filename",
        dest="filename",
        type=str,
        default="AstrometryAssessment",
        help="Filename",
    )

    args = parser.parse_args()
    if args.writer:
        writer(args)
    else:
        main(args)
