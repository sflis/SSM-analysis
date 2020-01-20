import numpy as np
from collections import namedtuple
import astropy.units as u
from numba import jit
from numba import vectorize, float64
import pickle
from collections import defaultdict, Counter
from ctapipe.coordinates import EngineeringCameraFrame, AltAz
from astropy.coordinates import SkyCoord  # AltAz, EarthLocation
from tqdm.auto import tqdm

import ssm
import os
from numpy import sin, cos
import daiquiri


def generate_hotspots(
    alt, az, altaz_frame, stars_pos, star_cat, pixsize, vmag_lim, pos
):

    telescope_pointing = SkyCoord(alt=alt, az=az, frame=altaz_frame)
    fov_mask = telescope_pointing.separation(stars_pos).deg < 5
    sind = np.argsort(star_cat.vmag.values[fov_mask])

    focal_length = u.Quantity(2.15191, u.m)

    engineering_frame = EngineeringCameraFrame(
        n_mirrors=2,
        location=telescope_pointing.location,
        obstime=telescope_pointing.obstime,
        focal_length=focal_length,
        telescope_pointing=telescope_pointing,
    )
    hotspots = []
    in_cat_stars = []
    hips = []
    for i, star in enumerate(stars_pos[fov_mask][sind]):
        star_en = star.transform_to(engineering_frame)
        p = np.array([star_en.x.value, star_en.y.value])
        if np.any(
            np.linalg.norm((pos - np.outer(p, np.ones(2048))).T, axis=1) < pixsize * 1.1
        ):
            for h in hotspots:
                if np.linalg.norm(np.array(h) - p) > pixsize * 4:
                    continue

            hotspots.append((star_en.x.value, star_en.y.value))
            hip = (i, star_cat.hip_number.values[fov_mask][sind][i])
            hips.append(hip)
            # print(hip, i)
            if star_cat.vmag.values[fov_mask][sind][i] < vmag_lim:
                in_cat_stars.append(hip)
    if len(in_cat_stars) == 0:
        print("no star in catalog for fov")
    # print("number of stars in catalog for fov", len(in_cat_stars))
    return hotspots, telescope_pointing, fov_mask, in_cat_stars, hips


root_dir = os.path.dirname(os.path.dirname(ssm.__file__))
StarPattern = namedtuple("StarPattern", "hip vmag eq_pos sp_pos sp_vmag sp_hip index")


class StarPatternMatch:

    """Summary

    Attributes:
        engineering_frame (TYPE): Description
        horizon_level (int): Description
        minpixdist (TYPE): Description
        obstime (TYPE): Description
        patterns (TYPE): Description
        pixsize (TYPE): Description
        star_coordinates (TYPE): Description
        star_table (TYPE): Description
        stars_above_horizon (TYPE): Description
    """

    def __init__(
        self,
        star_coordinates,
        star_table,
        patterns,
        pixsize,
        minpixdist,
        engineering_frame,
    ):
        self.star_table = star_table
        self.star_coordinates = star_coordinates
        self.patterns = patterns
        self.pixsize = pixsize
        self.minpixdist = minpixdist
        self.engineering_frame = engineering_frame
        self.horizon_level = 20
        self.obstime = engineering_frame.obstime
        altaz_frame = AltAz(
            location=self.engineering_frame.location, obstime=self.obstime
        )
        altaz_stars = self.star_coordinates.transform_to(altaz_frame)
        self.stars_above_horizon_ind = altaz_stars.alt.deg > self.horizon_level
        self.stars_above_horizon = self.star_table.hip_number.values[
            altaz_stars.alt.deg > self.horizon_level
        ]
        self.silence = False
        self.log = daiquiri.getLogger(__name__)

    @classmethod
    def from_location(
        clc,
        altaz_frame,
        stars,
        sdt,
        fov,
        focal_length,
        min_alt,
        pixsize,
        vmag_lim,
        minpixdist=5,
    ):
        log = daiquiri.getLogger(__name__)
        mask = sdt.vmag < vmag_lim
        sdt = sdt[mask]
        stars = stars[mask]

        path = os.path.join(
            root_dir,
            "caldata",
            "starcat_pixsize{:.3g}_fov{}_minpixdist{}_vmag{}.pkl".format(
                pixsize, fov, minpixdist, vmag_lim
            ),
        )
        if os.path.exists(path):
            with open(path, "rb") as f:
                try:
                    args = pickle.load(f)
                except Exception as e:
                    log.error(
                        "Error occured while loading star patterns from file.",
                        file=f,
                        error=e,
                    )
                    raise e
                return clc(**args)
        log.info(
            "No previously computed star pattern file found. Generating table now...."
        )
        star_patterns = {}

        for k, star in tqdm(enumerate(stars), total=len(stars)):

            telescope_pointing = SkyCoord(
                ra=star.ra, dec=star.dec, frame="icrs"  # ,altaz_frame,
            )
            tm_alt_az = telescope_pointing.transform_to(altaz_frame)
            engineering_frame = EngineeringCameraFrame(
                n_mirrors=2,
                location=altaz_frame.location,
                obstime=altaz_frame.obstime,
                focal_length=focal_length,
                telescope_pointing=tm_alt_az,
            )
            if tm_alt_az.alt.deg < min_alt:
                continue

            sep = telescope_pointing.separation(stars)
            ind = np.where(sep.deg < fov)[0]

            stars_eng = stars[ind].transform_to(engineering_frame)
            focus_star = stars_eng[np.where(ind == k)[0]]

            pf = np.array([focus_star.x.value[0], focus_star.y.value[0]])
            hipf = sdt.hip_number.values[k]
            vmagf = sdt.vmag.values[k]

            pattern = []
            for i, star1 in enumerate(stars_eng):
                if ind[i] == k:
                    continue
                hip = sdt.hip_number.values[ind][i]
                vmag = sdt.vmag.values[ind][i]
                pc = np.array([star1.x.value, star1.y.value])

                pattern.append((pf - pc, hip, vmag))

            pattern = sorted(pattern, key=lambda t: t[2])
            vmags = np.array([p[2] for p in pattern])
            hips = np.array([p[1] for p in pattern], dtype=np.uint32)
            pcs = np.array([p[0] for p in pattern])

            star_patterns[hipf] = StarPattern(
                hipf, vmagf, np.array([star.ra.rad, star.dec.rad]), pcs, vmags, hips, k
            )

        with open(path, "wb") as f:
            log.info("Generated star pattern table saved.", file=f)
            pickle.dump(
                dict(
                    star_coordinates=stars,
                    star_table=sdt,
                    patterns=star_patterns,
                    pixsize=pixsize,
                    minpixdist=minpixdist,
                    engineering_frame=engineering_frame,
                ),
                f,
            )

        return clc(stars, sdt, star_patterns, pixsize, minpixdist, engineering_frame)

    def recumpute_horizon(self, obstime, horizon_level):

        recumpute = False
        if obstime is not None and obstime != self.obstime:
            self.obstime = obstime
            recumpute = True
        if horizon_level is not None and horizon_level != self.horizon_level:
            self.horizon_level = horizon_level
            recumpute = True

        if recumpute:
            self.log.info("recomputing horizon and star selection")
            altaz_frame = AltAz(
                location=self.engineering_frame.location, obstime=self.obstime
            )
            altaz_stars = self.star_coordinates.transform_to(altaz_frame)
            self.stars_above_horizon_ind = altaz_stars.alt.deg > self.horizon_level
            self.stars_above_horizon = self.star_table.hip_number.values[
                self.stars_above_horizon_ind
            ]

    def identify_stars(
        self,
        hotspots,
        obstime=None,
        horizon_level=None,
        search_region=None,
        only_one_it=False,
    ):
        bins = np.linspace(-35 * self.pixsize * 2, 35 * self.pixsize * 2, 70 * 2)
        bins2d = (bins, bins)
        # center on the first hotspot
        shifted_hs = hotspots - hotspots[0]
        hotspotmap, n_hotspots, xm, ym = prepare_hotspotmap(shifted_hs[:10], bins2d)

        if n_hotspots < 2:
            self.log.warn(
                "Not enough hotspots to identify star patterns", n_hotspots=n_hotspots
            )
            return  # sky_map_missed[index] +=1

        self.recumpute_horizon(obstime, horizon_level)

        test_hips = self.stars_above_horizon
        if search_region is not None:
            point, fov = search_region
            sep = point.separation(self.star_coordinates[self.stars_above_horizon_ind])
            test_hips = self.star_table.hip_number.values[self.stars_above_horizon_ind][
                sep.deg < fov
            ]

        # First iteration
        match = []
        for (
            hip
        ) in test_hips:  # tqdm(test_hips, total=len(test_hips),disable=self.silence):
            pattern = self.patterns[hip]
            ret = matchpattern(hotspotmap, hotspots, xm, ym, pattern, bins2d, 6.1)
            if ret is not None:
                n_matched, n_stars = ret
                match.append(
                    (
                        hip,
                        n_matched,
                        n_matched / n_stars,
                        n_matched / n_hotspots,
                        pattern.sp_pos.shape[0],
                        n_hotspots,
                        n_stars,
                    )
                )

        if only_one_it:
            return match
        match = np.array(match)
        match_quantity = match[:, 2] * match[:, 3] * match[:, 1]
        m = np.argmax(match_quantity)
        if np.max(match_quantity) / np.mean(match_quantity) > 6:

            matchhip = match[m, 0]
            matched_hotspots = defaultdict(list)  # [(hotspots[0],matchhip,0)]
            matchedhips = defaultdict(list)
            test_hips = [int(matchhip)]  # list(self.patterns[int(matchhip)].sp_hip)

            for i in range(
                len(hotspots)
            ):  # tqdm(range(len(hotspots)), total=len(hotspots),disable=self.silence):
                shifted_hs = hotspots - hotspots[i]
                hotspotmap, n_hotspots, xm, ym = prepare_hotspotmap(shifted_hs, bins2d)
                match = []
                nmatch = []
                mhip = []
                for hip in test_hips:
                    pattern = self.patterns[hip]
                    ret = matchpattern_unbinned(shifted_hs, xm, ym, pattern, bins2d)
                    if ret is not None:
                        matched, n_stars, __init__ = ret
                        match.append(matched)
                        nmatch.append(len(matched))
                        mhip.append(hip)

                if len(match) > 0:
                    ind = np.argmax(nmatch)
                    matched_hotspots[i].append(mhip[ind])
                    matchedhips[mhip[ind]].append(i)
                    for m in match[ind]:
                        matched_hotspots[m[0]].append(m[1])
                        matchedhips[m[1]].append(m[0])
                    # print(f"Matched star  {mhip[ind]} for hotspot {i}")

                    test_hips = [m[1] for m in match[ind]]

            mhs = []
            for hs_i, hs in sorted(matched_hotspots.items()):
                c = Counter(hs)
                m = c.most_common()
                if m[0][1] > 3:
                    mhs.append((hotspots[hs_i], m[0][0], hs_i))

            for hs_i, hs in sorted(matchedhips.items()):
                c = Counter(hs)
                m = c.most_common()[0]
            if len(mhs) == 0:
                self.log.warn("No matching star pattern found.")
            return mhs
        else:
            self.log.warn("No matching star pattern found.")
            return None  # match

    def determine_pointing(self, matched_hotspots):
        hs = []
        spos = []
        hips = []
        for h in matched_hotspots:
            hs.append(h[0])
            hips.append(h[1])
            spos.append(
                self.star_coordinates[np.where(h[1] == self.star_table.hip_number)[0]]
            )
        # Shifting image center for first pointing approximation to most
        # center identified hotspot
        centermost_hs = np.argmin(np.linalg.norm(hs, axis=1))
        chs = hs[centermost_hs]
        shifted_hs = hs - chs
        cmsind = np.where(self.star_table.hip_number == hips[centermost_hs])[0]
        cmspos = self.star_coordinates[cmsind]
        # First pointing approximation
        cx, cy = solve_plate_constants(cmspos.ra.rad, cmspos.dec.rad, shifted_hs, spos)
        X, Y = plate2std(cx, cy, -chs[0], -chs[1])
        ra0, dec0 = std2eq(cmspos.ra.rad, cmspos.dec.rad, X, Y)

        # Second approximation using pointing from the first
        cx, cy = solve_plate_constants(ra0, dec0, hs, spos)
        X, Y = plate2std(
            cx, cy, 0, 0
        )  # Now the optical axis is assumed to be at the origin
        ra, dec = std2eq(ra0, dec0, X, Y)
        return float(ra), float(dec)


######################################################################
##########################Functions###################################
######################################################################

Hist1d = namedtuple("Hist1d", "bincontent binedges bincenters binwidths")
Hist2d = namedtuple("Hist2d", "bincontent binedges bincenters binwidths")


@jit(nopython=True, cache=True)
def hist1d(data, bins):
    """ Creates a 1D histogram of the provided data (accelerated with numba)

    Args:
        data (ndarray): 1D array containing the data to be histogrammed
        bins (ndarray): 1D array containing the binedges of the histogram

    Returns:
        Hist1d: A histogram object (namedtuple) containing the histogramed
                data and other fields describing the histogram

    """
    d_indices = np.digitize(data, bins)
    hist = np.zeros(bins.shape[0] - 1)
    for i in range(data.shape[0]):
        hist[d_indices[i] - 1] += 1
    binwidths = np.diff(bins)
    bincenters = bins[:-1] + binwidths / 2
    return Hist1d(
        bincontent=hist, binedges=bins, bincenters=bincenters, binwidths=[binwidths]
    )


@jit(nopython=True)
def hist2d(data, bins):
    """ Creates a 2D histogram of the provided data (accelerated with numba)

    Args:
        data (ndarray): 2D array containing the data to be histogrammed
        bins (ndarray): 2D array containing the binedges of the histogram

    Returns:
        Hist2d: A histogram object (namedtuple) containing the histogramed
                data and other fields describing the histogram

    """

    data = np.asarray(data)
    hist = np.zeros((bins[0].shape[0] - 1, bins[1].shape[0] - 1))
    d_indicesx = np.digitize(data[:, 0], bins[0])
    d_indicesy = np.digitize(data[:, 1], bins[1])
    for i in range(len(d_indicesy)):
        hist[d_indicesx[i] - 1, d_indicesy[i] - 1] += 1
    binedgesx, binedgesy = bins[0], bins[1]
    binwidths = [np.diff(binedgesx), np.diff(binedgesx)]
    bincentersx = binedgesx[:-1] + binwidths[0] / 2
    bincentersy = binedgesy[:-1] + binwidths[1] / 2
    return Hist2d(
        bincontent=hist,
        binedges=[binedgesx, binedgesy],
        bincenters=[bincentersx, bincentersy],
        binwidths=binwidths,
    )


@jit(nopython=True, cache=True)
def rotang(v1, v2=np.array([1, 0])):
    """Returns the angle between v1 and v2

    Args:
        v1 (array): The point on the plane to rotate from
        v2 (array, optional): The point on the plane to rotate to

    Returns:
        float: rotation angle in radians
    """
    return np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])


@jit(nopython=True, cache=True)
def rot_matrix(rot_ang):
    """ Returns a rotation matrix for the given
        an rotation angle `rot_ang`.

    Args:
        rot_ang (float): rotational angle

    Returns:
        array: rotation matrix
    """

    return np.array([[cos(rot_ang), -sin(rot_ang)], [sin(rot_ang), cos(rot_ang)]])


@jit(nopython=True, cache=True)
def prepare_hotspotmap(hs, bins):
    xm = (np.min(hs[:, 0]) - 0.01, np.max(hs[:, 0]) + 0.01)
    ym = (np.min(hs[:, 1]) - 0.01, np.max(hs[:, 1]) + 0.01)
    hotspotmap = hist2d(hs, bins)
    bw = hotspotmap.binwidths[0][0]
    # m = hotspotmap.bincontent > 1

    # hotspotmap.bincontent[m] = 1
    n_hotspots = np.sum(hotspotmap.bincontent)
    # spreading out the hotspots in their
    # neighboring bins
    for dx in np.arange(-bw, bw, 3):
        for dy in np.arange(-bw, bw, 3):
            tmpmap = hist2d(hs, bins)
            hotspotmap.bincontent[:] += tmpmap.bincontent
    m = hotspotmap.bincontent > 1
    tmpbinc = hotspotmap.bincontent.flatten()
    tmpbinc[m.flatten()] = 1
    hotspotmap.bincontent[:] = tmpbinc.reshape(hotspotmap.bincontent.shape)
    return hotspotmap, n_hotspots, xm, ym


@jit(nopython=True, cache=True)
def matchpattern(hotspotmap, hotspots, xlim, ylim, pattern, bins, vmag_lim=None):
    """Summary

    Args:
        hotspotmap (2-D array): Description
        hotspots (array): Description
        xlim (tuple): Description
        ylim (tuple): Description
        pattern (StarPattern): Description
        bins (2-D array): Description
        vmag_lim (None, optional): Description

    Returns:
        TYPE: Description

    """

    n_hotspots = np.sum(hotspotmap.bincontent)
    hotspot_i = np.where(hotspotmap.bincontent > 0)
    hs_x = hotspotmap.bincenters[0][hotspot_i[0]]
    hs_y = hotspotmap.bincenters[1][hotspot_i[1]]
    hotspots = np.empty((hs_x.shape[0], 2))
    hotspots[:, 0] = hs_x
    hotspots[:, 1] = hs_y

    vmag_lim = vmag_lim or np.inf
    vmag_starmask = np.where(pattern.sp_vmag < vmag_lim)[0]
    # if np.sum(vmag_starmask)< min_stars:
    #     sorted_ind_vmag = np.argsort(pattern.sp_vmag)
    #     vmag_starmask = sorted_ind_vmag[:min_stars]

    angs, star_mask, dbins, hs_bins, star_bins = prepare_pattern_rotations(
        hotspots, pattern.sp_pos[vmag_starmask]
    )

    m = np.zeros(angs.shape[0])
    n_stars = np.zeros(angs.shape[0])

    # Looping through all rotations and computing number of matches
    for i, n in enumerate(angs):

        rm = rot_matrix(n)
        rot_stars = np.dot(rm, pattern.sp_pos.T).T
        # We only care about stars that are within
        # a rectangle defined by the detected hotspots
        cam_fov_mask = (
            (rot_stars[:, 0] > xlim[0])
            & (rot_stars[:, 0] < xlim[1])
            & (rot_stars[:, 1] > ylim[0])
            & (rot_stars[:, 1] < ylim[1])
        )
        patternmap = hist2d(rot_stars[cam_fov_mask], bins)

        # Do not double count stars in the same bin
        # Maybe there is an easier way to index a ndarray in numba?
        hm = patternmap.bincontent > 1
        fh = patternmap.bincontent.flatten()
        fh[hm.flatten()] = 1
        patternmap.bincontent[:] = fh.reshape(patternmap.bincontent.shape)

        proj = hotspotmap.bincontent - patternmap.bincontent
        m[i] = np.sum(proj.flatten()[(proj > 0).flatten()])
        n_stars[i] = np.sum(cam_fov_mask)

    if len(m) > 0:
        minind = np.argmin(m)
        minm = m[minind]
        minn_stars = n_stars[minind]
        if minn_stars > 0:
            return n_hotspots - minm, minn_stars
        else:
            None
    else:
        return None


@jit(nopython=True, cache=True)
def matchpattern_unbinned(hotspots, xlim, ylim, pattern, bins, vmag_lim=None):
    """Summary

    Args:
        hotspots (array): Description
        xlim (tuple): Description
        ylim (tuple): Description
        pattern (StarPattern): Description
        bins (ndarray): 2D
        vmag_lim (None, optional): Description

    Returns:
        TYPE: Description

    """
    if vmag_lim is None:
        vmag_lim = np.inf
    vmag_starmask = pattern.sp_vmag < vmag_lim
    sel_star_pos = pattern.sp_pos[vmag_starmask]
    angs, star_mask, dbins, hs_bins, star_bins = prepare_pattern_rotations(
        hotspots, sel_star_pos
    )

    m = []
    n_matches = []
    n_stars = []
    m_angs = []
    # Looping through all rotations and computing number of matches
    for n in angs:
        rm = rot_matrix(n)

        rot_stars = np.dot(rm, sel_star_pos[star_mask].T).T
        # We only care about stars that are within
        # a rectangle defined by the detected hotspots
        cam_fov_mask = (
            (rot_stars[:, 0] > xlim[0])
            & (rot_stars[:, 0] < xlim[1])
            & (rot_stars[:, 1] > ylim[0])
            & (rot_stars[:, 1] < ylim[1])
        )

        matched_stars = []
        for bin_i in range(1, dbins.shape[0]):
            hs_bin_mask = (hs_bins == bin_i) | (hs_bins == (bin_i - 1))
            star_bin_mask = (star_bins == bin_i) | (star_bins == (bin_i - 1))

            for hs_ind in np.where(hs_bin_mask)[0]:
                hotspot = hotspots[hs_ind]
                for star_pos_ind in np.where(star_bin_mask & cam_fov_mask)[0]:
                    star_pos = rot_stars[star_pos_ind]
                    dist = np.linalg.norm(hotspot - star_pos)

                    if dist < 0.009:
                        matched_stars.append(
                            (
                                hs_ind,
                                pattern.sp_hip[vmag_starmask][star_mask][star_pos_ind],
                            )
                        )
        matched_stars = list(set(matched_stars))
        if len(matched_stars) > 0:
            m.append(matched_stars)
            n_matches.append(len(matched_stars))
            n_stars.append(np.sum(cam_fov_mask))
            m_angs.append(n)
    if len(m) > 0:
        minind = np.argmax(np.array(list(n_matches)))
        minm = m[minind]
        minn_stars = n_stars[minind]

        return minm, minn_stars, m_angs[minind]
    else:
        return None


@jit(nopython=True, cache=True)
def norm_row(arr):
    """ Computes the norm on a 2D array across the 2nd axis.

        This function is equivalent to np.linalg.norm(arr,axis=1),
        but unlike the mentioned function this function can be
        compiled with numba to be used in other functions that
        are compiled.
    Args:
        arr (array_like): a 2D array to be normed

    Returns:
        ndarray: A 1D array containing the norm of each row
                 in the input array arr.
    """
    norms = np.empty(arr.shape[0])
    for i in range(arr.shape[0]):
        norms[i] = np.linalg.norm(arr[i, :])
    return norms


@jit(nopython=True, cache=True)
def prepare_pattern_rotations(hotspots, star_pos):
    """ Finds all rotations of a star pattern that potentially matches at least one star.

        Determines which rotations of the star positions in
        star_pos will potentially match at least one of the hotspot
        positions in hotspots.

    Args:
        hotspots (array_like): hotspot positions
        star_pos (array_like): star positions in star pattern

    Returns:
        tuple: rotation angles for star patterns,
               mask of stars in the  pattern that can potentially match a hotspot,
               bins in the distance histogram,
               bin indices for the hotspots,
               bin indices for the stars in the star pattern
    """
    hs_dist = norm_row(hotspots)
    star_dist = norm_row(star_pos)
    dbins = np.arange(0, np.max(hs_dist) + 0.09, 0.006)

    rot_angs = []
    star_mask = star_dist < dbins[-1]
    star_bins = np.digitize(star_dist[star_mask], dbins) - 1
    hs_bins = np.digitize(hs_dist, dbins) - 1
    # We only want to find rotations that will potentially match
    # at least one star with one hotspot, therefore we only
    # rotate the pattern to hotspots that are at roghly the same
    # distance as some star in the pattern
    for bin_i in range(1, dbins.shape[0]):
        # Always looking at two neigboring bins
        hs_bin_mask = (hs_bins == bin_i) | (hs_bins == (bin_i - 1))
        star_bin_mask = (star_bins == bin_i) | (star_bins == (bin_i - 1))
        for i in range(hotspots[np.where(hs_bin_mask)[0]].shape[0]):
            hotspot = hotspots[np.where(hs_bin_mask)[0]][i, :]
            hotspot_rotang = rotang(hotspot)
            for j in range(star_pos[star_mask][np.where(star_bin_mask)[0]].shape[0]):
                spos = star_pos[star_mask][np.where(star_bin_mask)[0]][j, :]
                rot_angs.append(rotang(spos) - hotspot_rotang)

    angs = np.sort(
        np.array(list(set(rot_angs)))
    )  # using set to remove duplicate angles
    return angs, star_mask, dbins, hs_bins, star_bins


@jit(nopython=True, cache=True)
def std2eq(ra0: float, dec0: float, X: float, Y: float) -> tuple:
    """ Converts Standard plate coordinates to
        Equatorial coordinates

    Args:
        ra0 (float): pointing of the optical axis in right ascension (rad)
        dec0 (float): pointing of the optical axis in declination (rad)
        X (float): Standard plate x coordinate
        Y (float): Standard plate y coordinate

    Returns:
        tuple(float,float): coordinate pair (RA, DEC)
    """
    ra = ra0 + np.arctan(-X / (cos(dec0) - Y * sin(dec0)))
    dec = np.arcsin(sin(dec0) + Y * cos(dec0)) / np.sqrt(1.0 + X * X + Y * Y)
    return ra, dec


@jit(nopython=True, cache=True)
def eq2std(ra0: float, dec0: float, ra: float, dec: float) -> tuple:
    """ Converts Equatorial coordinates to Standard plate coordinates

    Args:
        ra0 (float): pointing of the plate optical axis in right ascension (rad)
        dec0 (float): pointing of the plate optical axis in declination (rad)
        ra (float):
        dec (float):

    Returns:
        tuple(float,float): coordinate pair (X,Y)
    """
    c = cos(dec0) * cos(dec) * cos(ra - ra0) + sin(dec0) * sin(dec)
    X = -(cos(dec) * sin(ra - ra0)) / c
    Y = -(sin(dec0) * cos(dec) * cos(ra - ra0) - cos(dec0) * sin(dec)) / c
    return X, Y


@jit(nopython=True, cache=True)
def plate2std(cx, cy, x, y):
    """ Converts plate coordinates to Standard plate
        coordinates

    Args:
        cx (array_like): Plate constants for x-axis
        cy (array_like): Plate constants for y-axis
        x (float): plate coordinate x
        y (float): plate coordinate y

    Returns:
        tuple(float,float): coordinate pair (X,Y)
    """
    X = cx[0] * x + cx[1] * y + cx[2]
    Y = cy[0] * x + cy[1] * y + cy[2]
    return X, Y


# @jit(nopython=True, cache=True)
def solve_plate_constants(ra0, dec0, hotspots, star_coordinates):
    """Summary

    Args:
        ra0 (float): Description
        dec0 (float): Description
        hotspots (array_like): Description
        star_coordinates (array_like): Description

    Returns:
        TYPE: Description

    """
    A = []
    for hs in hotspots:
        A.append([hs[0], hs[1], 1])
    A = np.array(A)
    yx = []
    yy = []
    for spos in star_coordinates:

        X, Y = eq2std(ra0, dec0, spos.ra.rad, spos.dec.rad)
        yx.append(X)
        yy.append(Y)

    yx = np.array(yx)
    yy = np.array(yy)

    cx = np.linalg.lstsq(A, yx, rcond=None)[0]
    cy = np.linalg.lstsq(A, yy, rcond=None)[0]
    return cx, cy
