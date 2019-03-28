import ssm
from ssm.core.sim_io import SimDataWriter, SourceDescr
from ssm.models import pixel_model
from ssm.simulation import sim_modules
from ssm.star_cat.hipparcos import load_hipparcos_cat
from ssm.models.calibration import RateCalibration
from ssm.core.pchain import ProcessingChain

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import datetime

from copy import copy

from os import path

ssm_path = path.dirname(ssm.__file__)


def main(
    start_time=Time("2019-05-01T04:00"),
    end_time=Time("2019-05-01T04:40"),
    time_step=datetime.timedelta(seconds=1),
    target=SkyCoord.from_name("vega"),
    pixm_file=path.join(ssm_path, "resources", "testpix_m.hkl"),
    calibration="SteveCalSmoothCutoff",
    output_file="VegaTrack40minFlatNSB.hdf",
):
    print("****Setting up simulation...")
    print("****Loading star catalog...")
    dt, stars = load_hipparcos_cat()
    sim_chain = ProcessingChain()
    time_step = datetime.timedelta(seconds=1)
    print("****Loading pixel response model from file")
    pixm = pixel_model.PixelResponseModel.from_file(pixm_file)
    t_mod = sim_modules.TimeGenerator(start_time, end_time, time_step)
    tel = sim_modules.Telescope(target=target)
    starproj = sim_modules.ProjectStars(dt, copy(stars))

    cal = RateCalibration(calibration)
    lig_prop = sim_modules.StarLightPropagator(pix_model=pixm)
    det_resp = sim_modules.DetectorResponse(calib=cal)
    nsb = sim_modules.FlatNSB(rate=40.0)
    writer = sim_modules.Writer(output_file)
    noise = sim_modules.Noise()
    print("****Building simulation chain")
    sim_chain.add(t_mod)
    sim_chain.add(tel)
    sim_chain.add(starproj)
    sim_chain.add(lig_prop)
    sim_chain.add(nsb)
    sim_chain.add(det_resp)
    sim_chain.add(noise)
    sim_chain.add(writer)
    print(sim_chain)

    print("****Configuring the modules, might take some time...")
    sim_chain.configure()
    print("****Starting simulation chain")
    sim_chain.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a slow signal simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        dest="start_time",
        type=str,
        default="2019-05-01T04:00",
        help="Start time of the simulation",
    )
    parser.add_argument(
        "-e",
        dest="end_time",
        type=str,
        default="2019-05-01T04:40",
        help="End time of the simulation",
    )
    parser.add_argument(
        "--step",
        dest="time_step",
        type=float,
        default=1.0,
        help="time step between frames in seconds",
    )

    parser.add_argument(
        "-t",
        "--target",
        dest="target",
        type=str,
        help="Target to track",
        default="vega",
    )
    parser.add_argument(
        "-r",
        dest="resp_par_path",
        type=str,
        default=path.join(ssm_path, "resources", "testpix_m.hkl"),
        help="path to response function parametrization",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        default="VegaTrack40minFlatNSB.hdf",
        help="Outputfile",
    )

    args = parser.parse_args()

    main(
        start_time=Time(args.start_time),
        end_time=Time(args.end_time),
        time_step=datetime.timedelta(seconds=args.time_step),
        target=SkyCoord.from_name(args.target),
        pixm_file=args.resp_par_path,
        calibration="SteveCalSmoothCutoff",
        output_file=arg.output,
    )
