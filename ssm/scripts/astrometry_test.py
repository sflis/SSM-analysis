import matplotlib.pyplot as plt
import numpy as np
import dashi

dashi.visual()

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from ctapipe.coordinates import EngineeringCameraFrame
from tqdm.auto import tqdm


from ssm.star_cat.hipparcos import load_hipparcos_cat
from CHECLabPy.plotting.camera import CameraImage

sdt, stars = load_hipparcos_cat()


# These will be the stars we see in data
source_star_table = sdt[sdt.vmag < 6.5]
source_stars = stars[sdt.vmag < 6.5]


# These are the stars we use to identify the patch of sky in
# the FOV
cvmag_lim = 6.1
catalog_star_table = sdt[sdt.vmag < cvmag_lim]
catalog_stars = stars[sdt.vmag < cvmag_lim]
# Define telescope frame
location = EarthLocation.from_geodetic(lon=14.974609, lat=37.693267, height=1750)
obstime = Time("2019-05-09T01:37:54.728026")
altaz_frame = AltAz(location=location, obstime=obstime)

# Get pixel coordinates from TargetCalib
from target_calib import CameraConfiguration

camera_config = CameraConfiguration("1.1.0")
mapping = camera_config.GetMapping()
xpix = np.array(mapping.GetXPixVector())  # * u.m
ypix = np.array(mapping.GetYPixVector())  # * u.m
pos = np.array([np.array(mapping.GetXPixVector()), np.array(mapping.GetYPixVector())])
size = mapping.GetSize() * u.m
area = size ** 2
pix_area = np.full(xpix.size, area) * area.unit
focal_length = u.Quantity(2.15191, u.m)
from ssm.pointing.astrometry import (
    rotang,
    rot_matrix,
    generate_hotspots,
    StarPatternMatch,
    matchpattern,
)

matcher = StarPatternMatch.from_location(
    altaz_frame=altaz_frame,
    stars=stars,
    sdt=sdt,
    fov=12,
    focal_length=focal_length,
    min_alt=0,
    vmag_lim=cvmag_lim,
    pixsize=mapping.GetSize(),
)


alt, az = np.deg2rad(13.21), 110  # 12.1
alt, az = np.deg2rad(73.21), 0  # 12.1
time_stamp ,alt,az = 1557390562.9098482, 0.25906850409532045, 0.7878222318029959
#1557351425.5583227, 1.120873732202405, 4.426301563444017
#1557354965.5608082, 1.2632359718811235, 4.477689919838186
#1557322263.4172812, 1.0130247059581006, 0.868233125943273
#1557362902.5210667, 1.1237636930813857, 4.602709354050866
#1557343327.5342753, 0.5482479003046035, 5.282641255134235
#1557379442.0362334, 0.015149983789129767, 2.9349089263799386
obstime = Time(time_stamp, format="unix")
altaz_frame = AltAz(location=location, obstime=obstime)
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

hotspots = true_hotspots.copy()
print(len(hotspots), len(hips_in_fov))
N_change = 1
# hotspots[N_change, :] = hotspots[N_change, :] + 0.003
check_matching_quantities = True
matched_hs = matcher.identify_stars(hotspots,
                                    obstime=obstime,
                                    horizon_level=0,
                                    only_one_it=check_matching_quantities)

if check_matching_quantities:
    match = np.array(matched_hs)
    match_quantity = match[:, 2] * match[:, 3] * match[:, 1]

    print(len(match))
    print(all_hips[0][1])
    print(np.where(match[:, 0] == all_hips[0][1])[0])
    index = np.where(match[:, 0] == all_hips[0][1])[0]
    plt.plot(match_quantity, 'ob')
    plt.plot(index, match_quantity[index], 'or')
    plt.title('match_quantity')
    plt.figure()
    plt.plot(match[:,1], 'ob')
    plt.plot(index, match[index,1], 'or')
    plt.title('N matched hotspots')
    plt.figure()
    plt.title('(matched hotspots)/(stars)')
    plt.plot(match[:,2], 'ob')
    plt.plot(index, match[index,2], 'or')
    plt.figure()
    plt.title('(matched hotspots)/(hotspots)')
    plt.plot(match[:,3], 'ob')
    plt.plot(index, match[index,3], 'or')
    # print(match[:, 2], match[:, 3], match[:, 1])
else:
    telescope_pointing = SkyCoord(alt=alt * u.rad, az=az * u.rad, frame=altaz_frame)
    print("True pointing:", telescope_pointing.transform_to("icrs"))
    if matched_hs is None or len(matched_hs) == 0:
        ra = dec = np.nan
        matched_hs = []
    else:
        ra, dec = matcher.determine_pointing(matched_hs)
    print("Estimated pointing:", np.rad2deg(ra), np.rad2deg(dec))
    fig, axs = plt.subplots(constrained_layout=True, figsize=(10, 6))
    # Different average camera images
    camera = CameraImage(xpix, ypix, mapping.GetSize(), ax=axs)
    camera.image = np.ones(2048)
    axs.plot(true_hotspots[0, 0], true_hotspots[0, 1], "ob")
    axs.plot(
        true_hotspots[:, 0],
        true_hotspots[:, 1],
        "o",
        color="gray",
        mfc="none",
        ms=25,
        mew=2,
    )
    for ths, hs in zip(true_hotspots, hotspots):
        if tuple(ths) != tuple(hs):
            axs.plot(hs[0], hs[1], "yo", mfc="none", ms=25, mew=3)
    for i, sid in enumerate(all_hips):
        # i = sid[0]
        hip = sid[1]
        plt.annotate(
            "{}".format(hip), (true_hotspots[i, 0], true_hotspots[i, 1]), color="r"
        )
    correct_id = 0
    wrong_id = 0
    for mhs in matched_hs:
        xy = mhs[0]
        # print(xy)
        textxy = xy - np.array([0.01, 0.01])
        plt.annotate(
            "{}".format(int(mhs[1])), (xy[0], xy[1]), (textxy[0], textxy[1]), color="w"
        )
        # print(all_hips[mhs[2]], mhs[2], int(mhs[1]))
        if all_hips[mhs[2]][1] != int(mhs[1]):
            axs.plot(
                hotspots[mhs[2], 0], hotspots[mhs[2], 1], "ro", mfc="none", ms=25, mew=1
            )
            wrong_id += 1
        else:
            axs.plot(
                hotspots[mhs[2], 0], hotspots[mhs[2], 1], "go", mfc="none", ms=25, mew=1
            )
            correct_id += 1
    telsky = telescope_pointing.transform_to("icrs")
    plt.title(
        "Number of hotspots {}, catalog stars {}, correct id {}, wrong id {}\n"
        "True pointing: {:.4f}° RA {:.4f}° DEC, \nEstimatied pointing: {:.4f}° RA, {:.4f}° DEC".format(
            len(hotspots),
            len(hips_in_fov),
            correct_id,
            wrong_id,
            telsky.ra.deg,
            telsky.dec.deg,
            np.rad2deg(ra),
            np.rad2deg(dec),
        )
    )
plt.show()
