import matplotlib.pyplot as plt
import numpy as np
import dashi
import healpy

dashi.visual()

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
# from ctapipe.coordinates import EngineeringCameraFrame
from tqdm.auto import tqdm

from ssdaq.data.io import FrameWriter
from ssdaq.data import Frame
from ssm.star_cat.hipparcos import load_hipparcos_cat
# from CHECLabPy.plotting.camera import CameraImage
from target_calib import CameraConfiguration
from ssm.pointing.astrometry import (
        generate_hotspots,
        StarPatternMatch,
    )

import zmq
def main(args):
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
    # xpix = np.array(mapping.GetXPixVector())  # * u.m
    # ypix = np.array(mapping.GetYPixVector())  # * u.m
    pos = np.array([np.array(mapping.GetXPixVector()), np.array(mapping.GetYPixVector())])
    # size = mapping.GetSize() * u.m
    # area = size ** 2
    # pix_area = np.full(xpix.size, area) * area.unit
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


    order = 10
    nside = healpy.order2nside(order)
    npix = healpy.nside2npix(nside)
    # sky_map_found = np.zeros(npix)
    npixs_above_horizon = np.where(healpy.pix2ang(nside,np.arange(npix))[0]>np.pi/2)[0]



    tstamp0 = 1557360406
    # writer = FrameWriter('AstrometryAssessment.icf')
    np.random.seed(args.seed)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    ip = '127.0.0.1'
    con_str = "tcp://%s:" % ip + str(args.port)
    socket.connect(con_str)
    for i in range(args.n_iterations):
        obstime = Time(tstamp0+np.random.uniform(-43200,43200),format='unix')
        altaz_frame = AltAz(location=location, obstime=obstime)
        # alt, az = np.deg2rad(73.21 ), 0#12.1
        pixid = int(np.random.uniform(0,npixs_above_horizon.shape[0]))
        ang = healpy.pix2ang(nside,pixid)
        alt = ang[0]
        az = ang[1]
        # az = np.random.uniform(0,2*np.pi)
        # alt = np.arccos(np.random.uniform(0,np.cos(np.pi*alt_min/180.)))
        hotspots, tel_pointing, star_ind, hips_in_fov, all_hips = generate_hotspots(
            alt * u.rad,
            az * u.rad,
            altaz_frame,
            source_stars,
            source_star_table,
            mapping.GetSize(),
            cvmag_lim,
            pos,
        )
        true_hotspots = np.array(hotspots)
        frame = Frame()
        frame.add('hips_in_fov', np.array(hips_in_fov))
        frame.add('hotspots', true_hotspots)
        tel_sky_pointing = tel_pointing.transform_to("icrs")
        frame.add('tel_pointing', np.array([tel_sky_pointing.ra.rad,tel_sky_pointing.dec.rad]))

        hotspots = true_hotspots.copy()
        N_change = 1
        hotspots[N_change, :] = hotspots[N_change, :] + 0.003

        matched_hs = matcher.identify_stars(hotspots, horizon_level=5, only_one_it=False)
        if matched_hs is not None and len(matched_hs)>0:
        # telescope_pointing = SkyCoord(alt=alt * u.rad, az=az * u.rad, frame=altaz_frame)
            ra, dec = matcher.determine_pointing(matched_hs)
            frame.add('matched_hs', np.array(matched_hs))
            frame.add('est_pointing', np.array([ra,dec]))

        socket.send(frame.serialize())

        # writer.write(frame)

#
    # writer.close()




import argparse

from ssdaq.core.basesubscribers import BasicSubscriber,WriterSubscriber
from ssdaq.data import io
class FrameSubscriber(BasicSubscriber):
    def __init__(self, ip: str, port: int, logger = None):
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
            **{k: v for k, v in locals().items() if k not in ["self", "__class__"]}
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
    parser.add_argument("-p, --port",dest='port', type=int,default=9999, help="port")
    parser.add_argument("-n, --n-iterations",dest='n_iterations', type=int,default=100, help="Number of iterations")
    parser.add_argument("-w, --writer",dest='writer',action='store_true',help='start writer')
    parser.add_argument('-s, --seed',dest='seed',type=int,default=1,help="random number generator seed")
    parser.add_argument("-f, --filename",dest='filename',type=str,default='AstrometryAssessment',help="Filename")

    args = parser.parse_args()
    if args.writer:
        writer(args)
    else:
        main(args)