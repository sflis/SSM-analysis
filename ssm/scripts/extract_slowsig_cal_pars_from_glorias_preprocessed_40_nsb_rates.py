import glob
import os
from collections import defaultdict
import pickle
import numpy as np
import dashi
dashi.visual()
import matplotlib.pyplot as plt
from ssm.calibration.slowcal import SlowSigCal
from ssm.utils.remotedata import load_data_factory
import ssm
# os.getcwd()
CAL_URL ="https://owncloud.cta-observatory.org/index.php/s/xMCCrNMOE2bPwwf"
root_dir=os.path.dirname(os.path.dirname(ssm.__file__))
files = ['KscRmr01dBnzch3','GVwAEQd9UaGxMKC','MzMG1W8Vxlhewzf']
s = r'https://owncloud.cta-observatory.org/index.php/s/%s/download'

load_datafuncs = [load_data_factory(s%(f),os.path.join(root_dir,'caldata',f),'40stepsNSB',lambda x:x) for f in files]
def readintensities(f):
    return np.array([float(r) for r in open(f,'r').readlines()])
loadintensities = load_data_factory("https://owncloud.cta-observatory.org/index.php/s/dcQWvybuoHQ2Eck/download",
                                    os.path.join(root_dir,'caldata','40_steps.txt'),
                                    'NSB_steps',
                                    readintensities)


def load_data(files):
        merged_data = defaultdict(lambda: defaultdict(list))
        # files = glob.glob(path)
        las_pos = [0]*2048
        for file in files:
            get_saved_file =pickle.load(open(file,'rb'))

            for k,v in get_saved_file.items():
                for pix,l in enumerate(v):
                    if k == "Position_peaks":
                        merged_data[k][pix] +=[las_pos[pix] + el for el in l]
                        if len(merged_data[k][pix])>0:
                            las_pos[pix] = merged_data[k][pix][-1]
                    else:
                        merged_data[k][pix] += l
            m = max(las_pos)+10000
            las_pos = [m]*2048
        print(merged_data.keys())
        return merged_data

def cal_report(cal):
    hist_a = dashi.histogram.hist1d(np.linspace(np.nanmin(cal.a),np.nanmax(cal.a),100))
    hist_a.fill(cal.a)
    hist_b = dashi.histogram.hist1d(np.linspace(np.nanmin(cal.b),np.nanmax(cal.b),100))
    hist_b.fill(cal.b)
    hist_c = dashi.histogram.hist1d(np.linspace(np.nanmin(cal.c),np.nanmax(cal.c),100))
    hist_c.fill(cal.c)
    thresh = (-cal.b + np.sqrt(cal.b**2-4*cal.a*cal.c))/(2*cal.a)
    hist_thresh = dashi.histogram.hist1d(np.linspace(np.nanmin(thresh),np.nanmax(thresh),100))
    hist_thresh.fill(thresh)

    plt.figure(figsize=(10,8))
    hist_a.line()
    hist_a.statbox()
    plt.title('a parameter')
    plt.savefig('plots/cal_report_a_par_hist.png')
    plt.figure(figsize=(10,8))
    hist_b.line()
    hist_b.statbox()
    plt.title('b parameter')
    plt.savefig('plots/cal_report_b_par_hist.png')
    plt.figure(figsize=(10,8))
    hist_c.line()
    hist_c.statbox()
    plt.title('c parameter')
    plt.savefig('plots/cal_report_c_par_hist.png')

    plt.figure(figsize=(10,8))
    hist_thresh.line()
    hist_thresh.statbox()
    plt.title('Threshold')
    plt.savefig('plots/cal_report_threshold_hist.png')

    from target_calib import CameraConfiguration

    cam_config = CameraConfiguration("1.1.0")
    mapping = cam_config.GetMapping()
    pixsize = mapping.GetSize()
    pix_posx = np.array(mapping.GetXPixVector())
    pix_posy = np.array(mapping.GetYPixVector())
    from CHECLabPy.plotting.camera import CameraImage
    f,a = plt.subplots(figsize=(10,8))
    camera = CameraImage(pix_posx, pix_posy,pixsize,ax=a)
    camera.image = cal.a
    camera.add_colorbar('')
    camera.ax.set_title('a parameter')
    camera.highlight_pixels(list(cal.badpixs['unphysical_cal']),color='r',linewidth=2)
    # camera.highlight_pixels(np.where(unphysical_rate)[0],color='b',linewidth=1.4)
    camera.highlight_pixels(list(cal.badpixs['no_cal']),color='k')
    camera.highlight_pixels(list(cal.badpixs['bad_fit']),color='b',linewidth=1.4)
    camera.set_limits_minmax(np.nanmin(cal.a),0)
    plt.savefig('plots/cal_report_a_par.png')

    f,a = plt.subplots(figsize=(10,8))
    camera = CameraImage(pix_posx, pix_posy,pixsize,ax=a)
    camera.image = cal.b
    camera.add_colorbar('')
    camera.ax.set_title('b parameter')
    camera.highlight_pixels(list(cal.badpixs['unphysical_cal']),color='r',linewidth=2)
    # camera.highlight_pixels(np.where(unphysical_rate)[0],color='b',linewidth=1.4)
    camera.highlight_pixels(list(cal.badpixs['no_cal']),color='k')
    camera.highlight_pixels(list(cal.badpixs['bad_fit']),color='b',linewidth=1.4)
    camera.set_limits_minmax(0,np.nanmax(cal.b))
    plt.savefig('plots/cal_report_b_par.png')

    f,a = plt.subplots(figsize=(10,8))
    camera = CameraImage(pix_posx, pix_posy,pixsize,ax=a)
    camera.image = cal.c
    camera.add_colorbar('')
    camera.ax.set_title('c parameter')
    camera.highlight_pixels(list(cal.badpixs['unphysical_cal']),color='r',linewidth=2)
    # camera.highlight_pixels(np.where(unphysical_rate)[0],color='b',linewidth=1.4)
    camera.highlight_pixels(list(cal.badpixs['no_cal']),color='k')
    camera.highlight_pixels(list(cal.badpixs['bad_fit']),color='b',linewidth=1.4)
    camera.set_limits_minmax(np.nanmin(cal.c),0)
    plt.savefig('plots/cal_report_c_par.png')




def main(diagnostic_plots=False,
        input_data_files=os.path.join(os.getcwd()+'/','output_40steps_*'),
        input_rate_steps_file='40_steps.txt',
        pixel_fit_config=None):
    """Summary

    Args:
        diagnostic_plots (bool, optional): Description
        input_data_files (TYPE, optional): Description
        input_rate_steps_file (str, optional): Description
        pixel_fit_config (None, optional): Description
    """

    intensities = loadintensities()#np.array([float(r) for r in open(input_rate_steps_file,'r').readlines()])
    files = [f() for f in load_datafuncs]
    return
    data = load_data(files)

    poshist = dashi.histogram.hist1d(np.linspace(0,600e3,1000))
    pos = data["Position_peaks"]
    ampl = data["Peak_val"]
    errors = data['Std_Error']
    print(errors)
    fit_points = defaultdict(dict)
    for pix, p in ampl.items():
        ampl[pix] = np.array(p)
        fit_points[pix]['amp'] = np.array(p)
        fit_points[pix]['stat_err'] = np.array(errors[pix])
    poss = set()
    if diagnostic_plots:
        plt.figure(figsize=(10,8))
    for i in range(2048):
        poshist.fill(np.array(pos[i]))
        poss = poss.union(set(pos[i]))
        if diagnostic_plots:
            plt.plot(pos[i],ampl[i],'o')

    if diagnostic_plots:
        plt.figure()
        poshist.line()

    #Histogramin illumination intervals for all pixels
    pos_el_mapping = {}
    prev_val = 0
    pos_el = 0
    for i, bc in enumerate(poshist.bincenters):
        cur_val = poshist.bincontent[i]
        if prev_val==0 and cur_val>0:
            pos_el_mapping[pos_el] = [bc-poshist.binwidths[i],None]
        if prev_val>0 and cur_val==0:
            pos_el_mapping[pos_el] = [pos_el_mapping[pos_el][0],bc+poshist.binwidths[i]]
            pos_el +=1
        prev_val = cur_val

    #Matching illumation intervals with illumation intensity
    intensity_indicies = defaultdict(list)
    for i, p  in pos.items():
        for pp in p:
            for ind,inter in pos_el_mapping.items():
                if pp >inter[0] and pp<inter[1]:
                    intensity_indicies[i].append(ind)
                    break
    for pix, inds in intensity_indicies.items():
        fit_points[pix]['nsb_rate'] = intensities[inds]

    #Storing different categories of bad pixels
    badpixs = {'unphysical_cal':set()}

    # Fitting 2nd degree polynomials
    params = {}
    print(pixel_fit_config)
    if pixel_fit_config is None:
        pixel_fit_config ={}
    for i in range(2048):
        if len(intensity_indicies[i])>0:
            mask = np.ones(len(ampl[i]), bool)
            deg = 2
            if i in pixel_fit_config:
                if 'pol_deg' in pixel_fit_config[i]:
                    deg = pixel_fit_config[i]['pol_deg']
                if 'excl' in pixel_fit_config[i]:
                 mask[pixel_fit_config[i]['excl']] = False

            fit_points[i]['mask'] = mask
            x,y,err = fit_points[i]['nsb_rate'],fit_points[i]['amp'],fit_points[i]['stat_err']
            p = np.polyfit(x[mask],y[mask],deg,full=True,w=1/(err[mask]+5)**2)
            params[i] = p
            fit_points[i]['params'] = p[0]
            fit_points[i]['residual'] = p[1]
            if p[0][0]>0 or p[0][1]<0 or p[0][2]>0:
                badpixs['unphysical_cal'].add(i)

    badpixs['no_cal'] = set(list(range(2048))).difference(set(params.keys()))
    badpixs['bad_fit'] = set()

    # Diagnistic plots for fit
    rate = np.linspace(0,3500,1000)
    for n,pixd in fit_points.items():
        if 'params' not in pixd:
            continue
        p = pixd['params']
        r = pixd['residual']
        badfit = r[0]>1.0
        if badfit or n in pixel_fit_config:
            fig = plt.figure(figsize=(10,8))
            y = np.polyval(p,rate)
            xdata,ydata = pixd['nsb_rate'],pixd['amp']
            mask = pixd['mask']
            m = np.abs(np.polyval(p,xdata[mask])-ydata[mask])/ydata[mask]>0.1
            plt.plot(xdata,ydata,'o',color='gray')
            plt.plot(xdata[mask],ydata[mask],'o')
            plt.plot(rate,y)
            plt.plot(xdata[mask][m],ydata[mask][m],'or')

            plt.ylim(1,3500)
            c,b,a = tuple(list(reversed(p))[:3])#p[0],p[1],p[2]
            thresh = (-b + np.sqrt(b**2-4*a*c))/(2*a)
            s = 'pixel {} \nresidual {:.2f} \nbadfit: {}\nparams:\n     a={:.3g}\n     b={:.3g}\n     c={:.3g}\n     thresh={:.3g}'.format(n,r[0],badfit,a,b,c,thresh)
            plt.text(5,500,s)
            plt.yscale('log')
            plt.xscale('log')
            for ind in np.arange(len(xdata))[mask][m]:
                plt.annotate("{}".format(ind),(xdata[ind],ydata[ind]))
            plt.ylabel('Amplitude (mV)')
            plt.xlabel('NSB eq. rate (MHz)')

            plt.savefig('plots/fitplot_diagnostics_log{}.png'.format(n))
            plt.close(fig)
        if badfit:
            badpixs['bad_fit'].add(n)
    for btype,bpxs in badpixs.items():
        print("Number of pixels with {}: {} ".format(btype,len(bpxs)))

    print(badpixs)
    with open('slow_cal.pkl','wb') as f:
        pickle.dump({'cal':params,'badpixs':badpixs},f)
    cal = SlowSigCal('slow_cal.pkl')
    cal_report(cal)


import yaml
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Start a simple Slow Signal readout listener.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs='?',
        type=str,
        help="Configfile",
    )
    args = parser.parse_args()
    config ={}
    if args.config:
        config = yaml.load(open(args.config,'rb'))
    main(**config)