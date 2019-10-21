import numpy as np
import astropy.units as u
from numba import jit
import pickle
# @jit(nopython=True)
def rotang(v1, v2=np.array([1, 0])):
    """Returns the angle between v1 and v2

    Args:
        v1 (array): The point on the plane to rotate from
        v2 (array, optional): The point on the plane to rotate to

    Returns:
        float: rotation angle in radians
    """
    return np.arctan2(v2[1], v2[0])-np.arctan2(v1[1], v1[0])

# @jit(nopython=True)
def rot_matrix(rot_ang):#v1, v2=np.array([1, 0])):
    """ Returns the a rotation matrix which rotates
        from v1 to v2.

    Args:
        rot_ang (float): rotational angle

    Returns:
        array: rotation matrix
    """

    return np.array(
        [[np.cos(rot_ang), -np.sin(rot_ang)],
         [np.sin(rot_ang), np.cos(rot_ang)]]
    )


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
            hip = (i,star_cat.hip_number.values[fov_mask][sind][i])
            hips.append(hip)
            print(hip,i)
            if star_cat.vmag.values[fov_mask][sind][i] < vmag_lim:
                in_cat_stars.append(hip)
    if len(in_cat_stars) == 0:
        print("no star in catalog for fov")
    print("number of stars in catalog for fov", len(in_cat_stars))
    print(len(hotspots),len(hips))
    print(hips)
    return hotspots, telescope_pointing, fov_mask, in_cat_stars,hips


import dashi
from ctapipe.coordinates import EngineeringCameraFrame, AltAz
from astropy.coordinates import SkyCoord  # AltAz, EarthLocation
from tqdm.auto import tqdm
from collections import namedtuple
import ssm
import os
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

    def __init__(self, star_coordinates, star_table, patterns, pixsize, minpixdist,engineering_frame):
        self.star_table = star_table
        self.star_coordinates = star_coordinates
        self.patterns = patterns
        self.pixsize = pixsize
        self.minpixdist = minpixdist
        self.engineering_frame = engineering_frame
        self.horizon_level = 20
        self.obstime = engineering_frame.obstime
        altaz_frame = AltAz(location=self.engineering_frame.location,
                          obstime=self.obstime)
        altaz_stars = self.star_coordinates.transform_to(altaz_frame)
        self.stars_above_horizon_ind = altaz_stars.alt.deg>self.horizon_level
        self.stars_above_horizon = self.star_table.hip_number.values[altaz_stars.alt.deg>self.horizon_level]

    @classmethod
    def from_location(
        clc, altaz_frame, stars, sdt, fov, focal_length, min_alt, pixsize,vmag_lim, minpixdist=5
    ):
        mask = sdt.vmag<vmag_lim
        sdt = sdt[mask]
        stars = stars[mask]

        path = os.path.join(root_dir, "caldata", "starcat_pixsize{:.3g}_fov{}_minpixdist{}_vmag{}.pkl".format(pixsize,fov,minpixdist,vmag_lim))
        if os.path.exists(path):
            with open(path,'rb') as f:
                args =pickle.load(f)
                return clc(**args)


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

            star_patterns[hipf] = StarPattern(hipf, vmagf, star, pcs, vmags, hips,k)


        with open(path,'wb') as f:
            pickle.dump(dict(star_coordinates = stars,
                             star_table = sdt,
                             patterns = star_patterns,
                             pixsize = pixsize,
                             minpixdist = minpixdist,
                             engineering_frame = engineering_frame,
                         ),
                        f
                        )

        return clc(stars, sdt, star_patterns, pixsize, minpixdist, engineering_frame)

    def identify_stars(self, hotspots, obstime=None, horizon_level=None, search_region=None):
        bins = np.linspace(-35*self.pixsize*2,35*self.pixsize*2,70*2)
        bins2d = (bins,bins)
        #center on the first hotspot
        shifted_hs = hotspots-hotspots[0]
        hotspotmap,n_hotspots,xm,ym = prepare_hotspotmap(shifted_hs,bins2d)

        if n_hotspots<2:
            return#sky_map_missed[index] +=1
        recumpute_horizon = False
        if obstime is not None and obstime!=self.obstime:
            self.obstime = obstime
            recumpute_horizon = True
        if horizon_level is not None and horizon_level!=self.horizon_level:
            self.horizon_level = horizon_level
            recumpute_horizon = True

        if recumpute_horizon:
            altaz_frame = AltAz(location=self.engineering_frame.location,
                                              obstime=self.obstime)
            altaz_stars = self.star_coordinates.transform_to(altaz_frame)
            self.stars_above_horizon_ind = altaz_stars.alt.deg>self.horizon_level
            self.stars_above_horizon = self.star_table.hip_number.values[self.stars_above_horizon_ind]

        test_hips = self.stars_above_horizon
        if search_region is not None:
            point,fov = search_region
            # print(point,fov)
            # print(self.stars_above_horizon_ind)
            # print(self.star_coordinates[self.stars_above_horizon_ind])
            sep = point.separation(self.star_coordinates[self.stars_above_horizon_ind])
            test_hips = self.star_table.hip_number.values[self.stars_above_horizon_ind][sep.deg < fov]

        #First iteration
        match = []
        for hip in tqdm(test_hips,total=len(test_hips)):
            pattern = self.patterns[hip]
            ret = matchpattern(hotspotmap,hotspots,xm,ym,pattern,bins2d)
            if ret is not None:
                n_matched,n_stars = ret
                match.append((hip,
                              n_matched,
                              n_matched/n_stars,
                              n_matched/n_hotspots,
                              pattern.sp_pos.shape[0],
                              n_hotspots,
                              n_stars
                              ))

        match = np.array(match)
        match_quantity = match[:,2]*match[:,3]*match[:,1]
        m =np.argmax(match_quantity)

        if np.max(match_quantity)/np.mean(match_quantity)>6:

            print("Matching")
            # i = np.where(self.star_table.hip_number.values== int(match[m,0]))
            # sep = self.star_coordinates[i].separation(self.star_coordinates)
            # ind = np.where(sep.deg < self.fov)[0]
            matchhip = match[m,0]
            matched_hotspots = [(hotspots[0],matchhip,0)]
            test_hips =self.patterns[int(matchhip)].sp_hip
            print(matchhip)
            print(test_hips)
            matched_hips = [matchhip]
            for i in tqdm(range(1,len(hotspots)),total=len(hotspots)-1):
                shifted_hs = hotspots-hotspots[i]
                hotspotmap,n_hotspots,xm,ym = prepare_hotspotmap(shifted_hs,bins2d)
                match = []
                for hip in test_hips:
                    if hip in matched_hips:
                        continue
                    pattern = self.patterns[hip]
                    ret = matchpattern(hotspotmap,hotspots,xm,ym,pattern,bins2d)
                    if ret is not None:
                        n_matched,n_stars = ret
                        match.append((hip,
                                      n_matched,
                                      n_matched/n_stars,
                                      n_matched/n_hotspots,
                                      n_hotspots,
                                      n_stars
                                      ))

                match = np.array(match)
                # print(match)
                m = (match[:,1]/np.max(match[:,1])>.5)&(match[:,2]>.6)
                match_quantity = match[:,2]*match[:,3]*match[:,1]
                m =np.argmax(match_quantity)
                print(match_quantity)
                print(match_quantity/np.mean(match_quantity))
                print(match[m,0])

                if np.max(match_quantity)/np.mean(match_quantity)>5:
                    print("Matched")
                    matchhip = match[m,0]
                    matched_hips.append(match[m,0])
                    matched_hotspots.append((hotspots[i],matchhip,i))
            return matched_hotspots
        else:
            print("No Match")
            return match
def prepare_hotspotmap(hs,bins):
        xm =(np.min(hs[:,0])-0.01,np.max(hs[:,0])+0.01)
        ym =(np.min(hs[:,1])-0.01,np.max(hs[:,1])+0.01)
        hotspotmap = dashi.histogram.hist2d(bins)
        bw = hotspotmap.binwidths[0][0]
        hotspotmap.fill(hs)
        m = hotspotmap.bincontent>1

        hotspotmap.bincontent[m] = 1
        n_hotspots = np.sum(hotspotmap.bincontent)
        # hs_in_map = np.array(
        # [hotspotmap.bincenters[0][m], hotspotmap.bincenters[1][m]]
        # ).T
        for dx in np.arange(-bw,bw,3):
            for dy in np.arange(-bw,bw,3):
                hotspotmap.fill(hs+np.array([dx,dy]))
        m = hotspotmap.bincontent>1
        hotspotmap.bincontent[m] = 1
        return hotspotmap,n_hotspots,xm,ym

def matchpattern(hotspotmap, hotspots, xlim, ylim, pattern, bins):

    angs = np.linspace(0, np.pi * 2, 360)
    n_hotspots = np.sum(hotspotmap.bincontent)
    hotspot_i = np.where(hotspotmap.bincontent > 0)

    hotspots = np.array(
        [hotspotmap.bincenters[0][hotspot_i[0]], hotspotmap.bincenters[1][hotspot_i[1]]]
    ).T
    hs_dist = np.linalg.norm(hotspots, axis=1)
    star_dist = np.linalg.norm(pattern.sp_pos, axis=1)
    dbins = np.linspace(0, np.max(star_dist) + 0.018, 24)

    hs_dist_hist = dashi.histogram.hist1d(dbins)
    star_dist_hist = dashi.histogram.hist1d(dbins)
    hs_dist_hist.fill(hs_dist)
    star_dist_hist.fill(star_dist)

    # Determining relevant rotations to match at least one hotspot with one star
    if np.sum(hs_dist_hist.bincontent * star_dist_hist.bincontent) < len(angs) * 0.7:
        rot_angs = []
        star_mask = star_dist < dbins[-1]
        star_bins = np.digitize(star_dist[star_mask], dbins)
        hs_bins = np.digitize(hs_dist, dbins)
        for bin_i in range(1, 24):
            hs_bin_mask = (hs_bins == bin_i) | (hs_bins == (bin_i - 1))
            star_bin_mask = (star_bins == bin_i) | (star_bins == (bin_i - 1))
            for hotspot in hotspots[np.where(hs_bin_mask)[0]]:
                for star_pos in pattern.sp_pos[star_mask][np.where(star_bin_mask)[0]]:
                    rot_angs.append(rotang(star_pos)-rotang(hotspot))
    angs = np.sort(np.array(list(set(rot_angs))))
    # minang = np.arctan2(0.006,0.2)
    # angs = angs[np.where(np.abs(np.diff(angs))>minang)[0]]

    bw = hotspotmap.binwidths[0][0]
    m = []
    n_stars = []
    # Looping through all rotations and computing number of matches
    for n in angs:
        patternmap = dashi.histogram.hist2d(bins)
        rm = rot_matrix(n)
        rot_stars = np.dot(rm, pattern.sp_pos.T).T
        cam_fov_mask = (
            (rot_stars[:, 0] > xlim[0])
            & (rot_stars[:, 0] < xlim[1])
            & (rot_stars[:, 1] > ylim[0])
            & (rot_stars[:, 1] < ylim[1])
        )
        # for dx in np.arange(-bw,bw,3):
        #     for dy in np.arange(-bw,bw,3):
        #         # hotspotmap.fill(hs+np.array([dx,dy]))
        patternmap.fill(
            rot_stars[cam_fov_mask]#+np.array([dx,dy])
        )
        patternmap.bincontent[patternmap.bincontent > 1] = 1
        proj = hotspotmap.bincontent - patternmap.bincontent
        s = np.sum(proj[proj > 0])

        if s < n_hotspots:
            m.append(s)
            n_stars.append(np.sum(cam_fov_mask))

    if len(m) > 0:
        minind = np.argmin(m)
        minm = m[minind]
        minn_stars = n_stars[minind]

        return n_hotspots - minm, minn_stars
    else:
        return None
