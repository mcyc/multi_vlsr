__author__ = 'mcychen'



"""
===================================================
Test the robustness of the multiple components fit
===================================================

^^^^^^^^^^
"""

import sys, os
# add the parent directory to the paths
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import pyspeckit
import astropy.io.fits as fits
from astropy import units as u
from pyspeckit.spectrum.units import SpectroscopicAxis, SpectroscopicAxes
from pyspeckit.spectrum.models.ammonia_constants import freq_dict
from pyspeckit.spectrum.models import ammonia_constants, ammonia
import pyspeckit.spectrum.models.ammonia_constants as nh3con
from pyspeckit.spectrum.units import SpectroscopicAxis as spaxis

from astropy import wcs, constants
from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar

import ammonia_multiv as ammv

import multi_v_fit as mvf
reload(mvf)

import master_fitter as mf
reload(mf)

#=======================================================================================================================


def make_wide_v_sep_cube():
    kwarg = {'version': 'lowC1_medC2', 'SNR1': 'low', 'SNR2': 'med'}
    #kwarg = {'version': 'medC1_highC2', 'SNR1': 'med', 'SNR2': 'high'}
    kwarg['mock_wide_v_sep'] = True
    kwarg['makeMockCube'] = True
    #kwarg['v1_extraoff'] = 1.75
    kwarg['v1_extraoff'] = 1.5
    run(linename="oneone", **kwarg)

def qm():
    l1 = "oneone"
    #kwarg = {'version':'lowC1_xlowC2', 'SNR1':'low', 'SNR2':'xlow'}
    #kwarg = {'version': 'medC1_medC2', 'SNR1': 'med', 'SNR2': 'med'}
    #kwarg = {'version': 'highC1_medC2', 'SNR1': 'high', 'SNR2': 'med'}
    #kwarg = {'version': 'medC1_highC2', 'SNR1': 'med', 'SNR2': 'high'}
    #run(l1, **kwarg)\
    '''
    kwarg = {'version': 'medC1_lowC2', 'SNR1': 'med', 'SNR2': 'low'}
    run(l1, **kwarg)
    '''

    kwarg = {'version': 'lowC1_medC2', 'SNR1': 'low', 'SNR2': 'med'}
    run(l1, **kwarg)
    kwarg = {'version': 'medC1_medC2', 'SNR1': 'med', 'SNR2': 'med'}
    run(l1, **kwarg)
    kwarg = {'version': 'highC1_medC2', 'SNR1': 'high', 'SNR2': 'med'}
    run(l1, **kwarg)
    kwarg = {'version': 'medC1_highC2', 'SNR1': 'med', 'SNR2': 'high'}
    run(l1, **kwarg)




def do():
    l1 = "oneone"
    l2 = "twotwo"

    kwarg = {'version':'lowC1_xxlowC2', 'SNR1':'low', 'SNR2':'xxlow'}
    run(l1, **kwarg)
    kwarg = {'version':'lowC1_xlowC2', 'SNR1':'low', 'SNR2':'xlow'}
    run(l1, **kwarg)
    kwarg = {'version':'lowC1_lowC2', 'SNR1':'low', 'SNR2':'low'}
    run(l1, **kwarg)
    kwarg = {'version':'medC1_medC2', 'SNR1':'med', 'SNR2':'med'}
    run(l1, **kwarg)
    kwarg = {'version':'medC1_lowC2', 'SNR1':'med', 'SNR2':'low'}
    run(l1, **kwarg)
    kwarg = {'version':'highC1_medC2', 'SNR1':'high', 'SNR2':'med'}
    run(l1, **kwarg)
    kwarg = {'version':'medC1_xlowC2', 'SNR1':'med', 'SNR2':'xlow'}
    run(l1, **kwarg)
    kwarg = {'version':'medC1_xxlowC2', 'SNR1':'med', 'SNR2':'xxlow'}
    run(l1, **kwarg)



def run(linename="oneone", version = "medC1_lowC2", SNR1="med", SNR2="low", recover_wide=False,
        v1_extraoff=None, makeMockCube=True, mock_wide_v_sep= True):
    #example_spec_fit()

    print("SNR1: {}, SNR2: {}.".format(SNR1, SNR2))

    baseDir = "/Users/mcychen/Documents/Data/GAS_NH3"




    if mock_wide_v_sep:
        paraDir = "{0}/mock_wideVSep_paraMaps/{1}".format(baseDir, version)
        cubeDir = "{0}/mock_wideVSep_rebase/{1}".format(baseDir, version)
    else:
        paraDir = "{0}/mock_paraMaps/{1}".format(baseDir, version)
        cubeDir = "{0}/mock_rebase/{1}".format(baseDir,  version)

    if not os.path.exists(cubeDir):
        os.makedirs(cubeDir)

    if not os.path.exists(paraDir):
        os.makedirs(paraDir)

    if linename == "oneone":
        line_root = "11"
        tex1, tau1 = mock_textau_11(SNR=SNR1)
        tex2, tau2 = mock_textau_11(SNR=SNR2)
    elif linename == "twotwo":
        line_root = "22"
        tex1, tau1 = mock_textau_22(SNR=SNR1)
        tex2, tau2 = mock_textau_22(SNR=SNR2)
    else:
        line_root = linename

    cubename = "{0}/mock_NH3_{1}_2vcomp_{2}_cube.fits".format(cubeDir, line_root, version)
    realparaname = "{0}/mock_NH3_{1}_2vcomp_{2}_trueparameter_maps.fits".format(paraDir, line_root, version)
    peakname = "{0}/mock_NH3_{1}_2vcomp_{2}_peaktemps.fits".format(cubeDir, line_root, version)


    if makeMockCube:
        # generate a fake cube
        print("tex1: {}, tau1: {}, tex2: {}, tau2: {}".format(tex1, tau1, tex2, tau2))
        kwargs = {'tex1':tex1, 'tau1':tau1, 'tex2':tex2, 'tau2':tau2}
        kwargs['v1_extraoff'] = v1_extraoff
        fake_cube(fname = cubename, paraname = realparaname, linename = linename, **kwargs)

    paraname = "{0}/mock_NH3_{1}_2vcomp_{2}_parameter_maps.fits".format(paraDir, line_root, version)
    modname = "{0}/mock_NH3_{1}_2vcomp_{2}_modelcube.fits".format(cubeDir, line_root, version)

    if False:
        # fit the fake cube with 2 velocity component models
        #pcube = example_cube_fit(cubename = cubename, paraname = paraname, modname = modname)
        #pcube = mvf.cubefit_gen(cube11name=cubename, ncomp=2, paraname=paraname, modname=modname, multicore = 3,
        #                    snr_min=1.0, linename=linename)
        region = mf.Region(cubePath=cubename, paraNameRoot="mock_NH3_11", paraDir=paraDir)
        mf.master_2comp_fit(region, snr_min=3, recover_wide=recover_wide)

    '''
    chiname = "{0}/mock_NH3_{1}_2vcomp{2}_chisq.fits".format(cubeDir, line_root, version)
    if False:
        cube = SpectralCube.read(cubename)
        model = fits.getdata(modname)
        return make_chi_tea(cubename, modname, savename = chiname)

    if False:
        return plot_para_check(realparaname, paraname, chiname, peakname)
    '''

    if False:
        figDir = "{0}/figures".format(paraDir)
        if not os.path.exists(figDir):
            os.makedirs(figDir)
        plot_vel_fit_accuracy(realparaname, paraname, saveFigDir=figDir, saveFigRoot="NH3_{0}".format(line_root))

    #return region


def plot_vel_fit_accuracy(name_realp, name_fitp, saveFigDir="",  saveFigRoot=""):

    import matplotlib.pyplot as plt

    para_rl, hdr_rl = fits.getdata(name_realp, header=True)
    para_ft, hdr_ft = fits.getdata(name_fitp, header=True)


    '''
    # if the first component model fits the real second component better, swap it with the second component
    # note: this may be an issue for cases where two components have a very similar velocity
    swap = np.abs(para_ft[0] - para_rl[2]) < np.abs(para_ft[4] - para_rl[2])
    # note: check to see if the following operation does "overide" parts of the information
    para_ft[:4][:,swap], para_ft[4:8][:,swap]  = para_ft[4:8][:,swap], para_ft[:4][:,swap]
    '''

    # plot the 'real' error of the second component fit

    if False:
        # the fit error vs "real" error
        plt.clf()
        plt.scatter(para_ft[12], np.abs(para_ft[4] - para_rl[2]), s=3)
        plt.xlabel("fits error")
        plt.ylabel("real error")

    if False:
        # accuracy of the vlsr fit vs. vlsr seperation from the bright component
        plt.clf()
        plt.scatter(para_rl[2], para_ft[4] - para_rl[2], s=3)
        plt.xlabel(r"$\Delta$v$_{lsr}$ between the two components (km s$^{-1}$)")
        plt.ylabel(r"Fit and actual v$_{lsr}$ difference (km s$^{-1}$)")
        plt.savefig("{0}/{1}_vlsrErr_vs_deltaV_scatter.pdf".format(saveFigDir, saveFigRoot))

    if False:
        # accuracy of the vlsr fit vs. vlsr seperation from the bright component
        plt.clf()
        plt.scatter(para_rl[3], para_ft[4] - para_rl[2], s=3)
        plt.xlabel(r"Second Component $\sigma_{v}$ (km s$^{-1}$)")
        plt.ylabel(r"Difference between fit and actual v$_{lsr}$ (km s$^{-1}$)")
        plt.savefig("{0}/{1}_vlsrErr_vs_deltaSigma_scatter.pdf".format(saveFigDir, saveFigRoot))

    if False:
        # plot the histogram of the fit errors
        plt.clf()
        plt.hist((para_ft[0] - para_rl[0]).ravel(), 50, range=(-0.4,0.4), normed=False, histtype = "stepfilled", color="0.75")
        plt.hist((para_ft[4] - para_rl[2]).ravel(), 50, range=(-0.4,0.4), normed=False, histtype = "step")
        plt.legend(["rear component","front component"], frameon=False)
        plt.ylabel("Number of pixels")
        plt.xlabel(r"Difference between fit and actual v$_{lsr}$ (km s$^{-1}$)")
        plt.savefig("{0}/{1}_vlsrErr_histo.pdf".format(saveFigDir, saveFigRoot))

    if True:
        # plot the histogram of the fit errors relative to the estimated errors
        plt.clf()
        diff1 = (para_ft[0] - para_rl[0])/para_ft[8]
        diff2 = (para_ft[4] - para_rl[2])/para_ft[12]
        plt.hist(diff1.ravel(), 50, range=(-5,5), normed=False, histtype = "stepfilled", color="0.75")
        plt.hist(diff2.ravel(), 50, range=(-5,5), normed=False, histtype = "step")
        plt.legend(["rear component","front component"], frameon=False)
        plt.title("Accuracy in the 1,1 fits")
        plt.ylabel("Number of pixels")
        plt.xlabel(r"Difference between fit and actual v$_{lsr}$ over the estimated error")
        plt.savefig("{0}/{1}_vlsrErrRelEst_histo.pdf".format(saveFigDir, saveFigRoot))

    #plt.show()


def plot_para_check(name_realp, name_fitp, name_chi, peaktname):
    # check to see how well the fitting routine performed
    para_rl, hdr_rl = fits.getdata(name_realp, header=True)
    para_ft, hdr_ft = fits.getdata(name_fitp, header=True)
    chi, hdr_chi = fits.getdata(name_chi, header=True)
    peakT, hdr_peakT = fits.getdata(peaktname, header=True)
    noise = 0.15 # K km/s; typical of GAS NH3 (1-1) data
    snr = peakT/noise


    swap = para_ft[0] > para_ft[4]
    t = np.copy(para_ft)
    para_ft[:4][:,swap] = para_ft[4:8][:,swap]
    para_ft[4:8][:,swap] = t[:4][:,swap]

    #para_ft[:8][:, swap], para_ft[8:][:, swap] = para_ft[8:][:, swap], para_ft[:8][:, swap]

    #para_ft[0, swap], para_ft[4, swap] = para_ft[4, swap], para_ft[0, swap]
    #para_ft[1, swap], para_ft[5, swap] = para_ft[5, swap], para_ft[1, swap]

    import matplotlib.pyplot as plt
    plt.clf()
    '''
    plt.scatter(para_rl[2], para_ft[0], zorder = 20, s=1)
    plt.scatter(para_rl[2], para_ft[4], zorder = 15, s=1)
    plt.plot(para_rl[2], para_rl[2], c="0.5", zorder = 5)
    '''

    '''
    plt.scatter(para_rl[3], para_ft[1], zorder = 20, s=1)
    plt.scatter(para_rl[3], para_ft[5], zorder = 15, s=1)
    plt.plot(para_rl[3], para_rl[3], c="0.5", zorder = 5)
    '''

    if False:
        #plt.scatter(para_rl[2], para_rl[2] - para_ft[4], zorder = 20, s=1, c=chi, vmin=0.8, vmax=1.3)
        #plt.colorbar()

        plt.scatter(snr[1], para_rl[2] - para_ft[4], zorder = 20, s=1, c=chi, vmin=0.8, vmax=1.3)
        plt.colorbar(label = r'$\chi_{\nu}$')

        #plt.scatter(para_rl[2], para_rl[0] - para_ft[0], zorder = 20, s=1)
        #plt.xlabel(r"Real v$_{lsr}$ (km s$^{-1}$)")
        print snr[0][0,0]
        plt.annotate('SNR is: {0}'.format(int(snr[0][0,0])), xy=(0.5, 0.5))
        plt.title(r"Fitted v$_{lsr}$ of the second component")
        plt.xlabel(r"Peak signal-to-noise ratio")
        plt.ylabel(r"Real v$_{lsr}$ - Fitted v$_{lsr}$ (km s$^{-1}$)")

    if True:
        gmask = snr[1] > 3.0
        plt.hist((para_rl[2] - para_ft[4])[gmask].ravel(), 100, range=(-0.2,0.2), normed=True, histtype = "step")
        plt.hist((para_rl[0] - para_ft[0]).ravel(), 50, range=(-0.2,0.2), normed=True, histtype = "step")


    #plt.scatter(para_rl[3], para_rl[2] - para_ft[4], zorder = 20, s=1)
    plt.show()

    return para_rl, para_ft





def make_chi_tea(cubename, modname, savename = None):
    # calculate chi-squred map based on input models
    cube = SpectralCube.read(cubename)
    model = fits.getdata(modname)
    chisq = chi_sq_map(cube, model)
    if savename != None:
        hdu = fits.PrimaryHDU(chisq, cube.wcs.celestial.to_header())
        hdu.writeto(savename, overwrite=True)

    return chisq

def chi_sq_map(cube, model, expand=20):
    '''
    cube : SpectralCube

    model: numpy array

    expand : int
        Expands the region where the residual is evaluated by this many channels in the spectral dimension
    '''

    import scipy.ndimage as nd
    #model = np.zeros(cube.shape)
    cube = cube.with_spectral_unit(u.Hz, rest_value = freq_dict['oneone']*u.Hz)

    mask = model > 0
    residual = cube.filled_data[:].value-model

    # This calculates chisq over the region where the fit is non-zero
    # plus a buffer of size set by the expand keyword.

    selem = np.ones(expand,dtype=np.bool)
    selem.shape += (1,1,)
    mask = nd.binary_dilation(mask, selem)
    mask = mask.astype(np.float)
    chisq = np.sum((residual * mask)**2, axis=0) / np.sum(mask, axis=0)

    # This produces a robust estimate of the RMS along every line of sight:
    diff = residual - np.roll(residual, 2, axis=0)
    rms = 1.4826 * np.nanmedian(np.abs(diff), axis=0) / 2**0.5

    chisq /= rms**2

    return chisq



def generate_xarr(linename):
    # generate SpectroscopicAxis objects
    # the -1.0 term in the end is to insure the channel is in the increasing order in frequency space, consistent
    # with GAS data
    channelwidth = (5.72 * u.kHz / (nh3con.freq_dict[linename] * u.Hz)) * constants.c * -1.0

    xarr = spaxis(np.arange(-500, 500) * channelwidth,
                  unit='GHz',
                  refX=nh3con.freq_dict[linename] / 1e9,
                  velocity_convention='radio', refX_unit='GHz')
    return xarr




def fake_cube(fname=None, paraname=None, linename="oneone", map_shape=(20,20), sigm1_ul=0.3, v1_maxoff=0.5, v1_extraoff=None, rms=0.1,
              **kwargs):
    # Create a fake spectral ammonia (1-1) cube with GAS spectral resolution
    # rms of 0.1 is roughly the GAS level

    # note: the largest linewidth seen in GAS DR1 is ~1.4 in Orion A, the rest 3 regions all have sigma < 1.0 km/s
    # the sigm1_ul and v1_maxoff default values used here are thus based on a naive guess on what the typical GAS
    # values are based on these DR1, single component fit results

    peakname = None

    '''
    # Create a new WCS object so we can instantiate the SpectralCube
    mywcs = wcs.WCS(naxis=3)

    # GAS spectral resolution in km/s
    spc_res = 0.07

    # Spectral range of the cube
    spc_ulim = 30.0
    spc_llim = -30.0
    n_samp = (spc_ulim - spc_llim)/spc_res

    # Set up a tangent projection (which is normally read from a file.)
    mywcs.wcs.crpix = [1,1,1]
    pixwidth = 0.0024440485689343
    mywcs.wcs.cdelt = np.array([-pixwidth, pixwidth, spc_res*1e3])
    mywcs.wcs.crval = [290.9250, 14.5092, spc_llim*1e3]
    mywcs.wcs.ctype = ["RA---TAN", "DEC--TAN", 'VELO']
    mywcs.wcs.cunit = ['deg', 'deg', 'm/s']

    # Create a synthetic X-dimension in km/s
    xarr = np.linspace(spc_llim, spc_ulim, int(n_samp) + 1, endpoint = True)
    xarr = SpectroscopicAxis(xarr*u.km/u.s, velocity_convention='radio', refX=freq_dict[linename]*u.Hz).as_unit(u.GHz)
    '''
    xarr = generate_xarr(linename)

    # First velocity component
    sigm0 = 0.1
    vlsr0 = 0.0

    # upper and lower limits of the second velocity component
    # the 2nd componet sigma and vlsr will be incremented in these ranges
    sigm1_ll = sigm0
    vlsr1_ul = vlsr0 + v1_maxoff
    vlsr1_ll = vlsr0

    if not v1_extraoff is None:
        vlsr1_ul = vlsr1_ul + v1_extraoff
        vlsr1_ll = vlsr1_ll + v1_extraoff

    # the grid dimension (sigma1, vlsr1)
    nx, ny = map_shape

    # now create the grid
    sigm1 = np.linspace(sigm1_ll, sigm1_ul, nx)
    vlsr1 = np.linspace(vlsr1_ll, vlsr1_ul, ny)

    #data = np.empty((int(n_samp) + 1, ny, nx))
    data = np.zeros((xarr.shape[0], ny, nx))
    peakT = np.empty((2, ny, nx))

    # rough estimate from Radex
    # component one: assuming the excitation condition of Tkin = 10K,  n = 1e5 cm^-3, N = 3e14 cm^-2, "sigma" = 0.2
    # component two: assuming the excitation condition of Tkin = 10K,  n = 1e4 cm^-3, N = 1e13 cm^-2, and a "sigma"
    # that produces a peak signal of ~0.5 K

    tex1 = 6.0
    tau1 = 2.0
    tex2 = 3.5
    tau2 = 0.4

    if linename == "twotwo":
        tex1 = 8.9
        tau1 = 3.9
        tex2 = 5.6
        tau2 = 0.42
    else:
        if linename != "oneone":
            print "[WARNING]: assuming excitation conditions for 1-1 line if no further user inputs are specified"

    if 'tex1' in kwargs:
        tex1 = kwargs['tex1']
    if 'tau1' in kwargs:
        tau1 = kwargs['tau1']
    if 'tex2' in kwargs:
        tex2 = kwargs['tex2']
    if 'tau2' in kwargs:
        tau2 = kwargs['tau2']

    print "kwargs: ".format(kwargs)
    print tex1, tau1, tex2, tau2


    sigm1_v, vlsr1_v = np.meshgrid(sigm1, vlsr1)
    sigm0_v = np.ones(map_shape) * sigm0
    vlsr0_v = np.ones(map_shape) * vlsr0

    tex0_v = np.ones(map_shape) * tex1
    tex1_v = np.ones(map_shape) * tex2

    tau0_v = np.ones(map_shape) * tau1
    tau1_v = np.ones(map_shape) * tau2

    xmat, ymat = np.indices(map_shape)
    for xx, yy in ProgressBar(zip(xmat.flatten(), ymat.flatten())):
        data[:, yy, xx] = ammv.ammonia_multi_v(xarr, vlsr0_v[yy, xx], sigm0_v[yy, xx], tex0_v[yy, xx], tau0_v[yy, xx],
                                               vlsr1_v[yy, xx], sigm1_v[yy, xx], tex1_v[yy, xx], tau1_v[yy, xx],
                                               line_names=[linename])

    # now add noise
    data += np.random.randn(*data.shape) * rms

    '''
    # ammonia.ammonia doens't take numpy array, so I'm just going to loop through the values....
    for i, sig in enumerate(sigm1):
        for j, v in enumerate(vlsr1):
            #c1 = ammonia.ammonia(xarr, trot=10, ntot=14.5, fortho=0.5, xoff_v=vlsr0, width=sigm0)
            #c2 = ammonia.ammonia(xarr, trot=20, ntot=14, fortho=0.5, xoff_v=v, width=sig)
            #spectrum = c1 + c2
            spectrum = ammv.ammonia_multi_v(xarr, vlsr0, sigm0, tex1, tau1, v, sig, tex2, tau2, line_names=[linename])
            data[:,j,i] = spectrum
            data[:,j,i] = spectrum
            #peakT[0,j,i], peakT[1,j,i] = c1.max(), c2.max()
    '''

    # now add noise
    #stddev = rms                # K km/s; typical of GAS NH3 (1-1) data
    #noise = np.random.normal(loc=0.0, scale=stddev, size=np.shape(data))
    #data = data + noise

    '''
    hdr = fits.Header()
    beamsize = 0.008554 # GAS Beam size
    hdr.set('BMAJ', beamsize)
    hdr.set('BMIN', beamsize)
    hdr.set('BPA',0)
    '''

    hdrkwds = {'BUNIT': 'K',
               'INSTRUME': 'KFPA    ',
               'BMAJ': 0.008554169991270138,
               'BMIN': 0.008554169991270138,
               'TELESCOP': 'GBT',
               'WCSAXES': 3,
               'CRPIX1': 2,
               'CRPIX2': 2,
               'CRPIX3': 501,
               'CDELT1': -0.008554169991270138,
               'CDELT2': 0.008554169991270138,
               'CDELT3': 5720.0,
               'CUNIT1': 'deg',
               'CUNIT2': 'deg',
               'CUNIT3': 'Hz',
               'CTYPE1': 'RA---TAN',
               'CTYPE2': 'DEC--TAN',
               'CTYPE3': 'FREQ',
               'CRVAL1': 0.0,
               'CRVAL2': 0.0,
               'LONPOLE': 180.0,
               'LATPOLE': 0.0,
               'EQUINOX': 2000.0,
               'SPECSYS': 'LSRK',
               'RADESYS': 'FK5',
               'SSYSOBS': 'TOPOCENT'}


    hdu = fits.PrimaryHDU(data)
    for kk in hdrkwds:
        hdu.header[kk] = hdrkwds[kk]

    #hdu.header['TMAX'] = Tmax
    #hdu.header['TMAX-1'] = Tmax_a
    #hdu.header['TMAX-2'] = Tmax_b
    #hdu.header['RMS'] = noise_rms
    hdu.header['CRVAL3'] = nh3con.freq_dict[linename]
    hdu.header['RESTFRQ'] = nh3con.freq_dict[linename]

    if fname != None:
        hdu.writeto(fname, overwrite=True)

    '''
    cube = SpectralCube(data=data, wcs=mywcs, header=hdr)
    cube = cube.with_spectral_unit(xarr.unit, rest_value = freq_dict[linename]*u.Hz)
    cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio')

    if fname != None:
        cube.write(fname, format='fits', overwrite = True)
    '''

    # the write the fake parameters into a cube
    if paraname != None:
        sigm1_v, vlsr1_v = np.meshgrid(sigm1, vlsr1)
        sigm0_v = np.ones(np.shape(sigm1_v))*sigm0
        vlsr0_v = np.ones(np.shape(sigm1_v))*vlsr0

        tex0_v = np.ones(np.shape(sigm1_v))*tex1
        tex1_v = np.ones(np.shape(sigm1_v))*tex2

        tau0_v = np.ones(np.shape(sigm1_v))*tau1
        tau1_v = np.ones(np.shape(sigm1_v))*tau2

        data = np.array([vlsr0_v, sigm0_v, tex0_v, tau0_v, vlsr1_v, sigm1_v, tex1_v, tau1_v])
        #fitcubefile = fits.PrimaryHDU(data=data, header=cube.header)
        fitcubefile = fits.PrimaryHDU(data=data, header=hdu.header)

        fitcubefile.header.set('CUNIT1', 'deg')
        fitcubefile.header.set('CUNIT2', 'deg')

        fitcubefile.header.set('PLANE1','VELOCITY_0')
        fitcubefile.header.set('PLANE2','SIGMA_0')
        fitcubefile.header.set('PLANE3', 'TEX_0')
        fitcubefile.header.set('PLANE4', 'TAU_0')

        fitcubefile.header.set('PLANE5','VELOCITY_1')
        fitcubefile.header.set('PLANE6','SIGMA_1')
        fitcubefile.header.set('PLANE7', 'TEX_1')
        fitcubefile.header.set('PLANE8', 'TAU_1')

        fitcubefile.header.set('CDELT3',1)
        fitcubefile.header.set('CTYPE3','FITPAR')
        fitcubefile.header.set('CRVAL3',0)
        fitcubefile.header.set('CRPIX3',1)
        fitcubefile.header.set('NAXIS3',4)
        fitcubefile.writeto(paraname ,overwrite=True)

    if peakname != None:
        #fitcubefile = fits.PrimaryHDU(data=peakT, header=cube.header)
        fitcubefile = fits.PrimaryHDU(data=peakT, header=hdu.header)
        fitcubefile.header.set('PLANE1','PEAK_0')
        fitcubefile.header.set('PLANE2','PEAK_0')
        fitcubefile.header.set('CDELT3',1)
        fitcubefile.header.set('CTYPE3','FITPAR')
        fitcubefile.header.set('CRVAL3',0)
        fitcubefile.header.set('CRPIX3',1)
        fitcubefile.header.set('NAXIS3',2)
        fitcubefile.writeto(peakname ,overwrite=True)

    #return cube


def mock_textau_11(SNR="low"):
    # all the tex and tau are calculated based on Radex on-line (van der Tak)
    # note, the Tpeak reported here are half of those calculated by Radex, since Radex does not consider hyperfine structures
    # and hence

    if SNR == "xxlow":
        # Tpeak ~0.13 K (below GAS rms noise of 0.1 K)
        # assuming the excitation condition of Tkin = 10K,  n = 1e3 cm^-3, N = 1e13 cm^-2, "sigma" = 1.0
        tex = 3.56
        tau = 0.39

    elif SNR == "xlow":
        # Tpeak ~0.30 K
        # assuming the excitation condition of Tkin = 10K,  n = 2e3 cm^-3, N = 3e13 cm^-2, "sigma" = 0.6
        tex = 3.7
        tau = 0.94

    elif SNR == "low":
        # Tpeak ~0.5 K
        # assuming the excitation condition of Tkin = 10K,  n = 2e3 cm^-3, N = 3e13 cm^-2, "sigma" = 0.3
        tex = 3.9
        tau = 1.8

    elif SNR == "med":
        # Tpeak ~1.1 K
        # assuming the excitation condition of Tkin = 10K,  n = 3e3 cm^-3, N = 5e13 cm^-2, "sigma" = 0.2
        tex = 5.0
        tau = 3.5

    elif SNR == "high":
        # Tpeak ~2.06 K
        # assuming the excitation condition of Tkin = 10K,  n = 5e3 cm^-3, N = 1e14 cm^-2, "sigma" = 0.15
        # note: this may cause problems with auto window guessing when self-absorption casues main hyperfine components
        # to have lower antenna temperature than the satellites
        tex = 6.8
        tau = 6.8

    elif SNR == "xhigh":
        # Tpeak ~3.1 K
        # assuming the excitation condition of Tkin = 10K,  n = 1e4 cm^-3, N = 1e14 cm^-2, "sigma" = 0.1
        # note: this may cause problems with auto window guessing when self-absorption casues main hyperfine components
        # to have lower antenna temperature than the satellites
        tex = 9.0
        tau = 7.8
    else:
        print "[ERROR] the passed in SNR value is not recognized!"
        return None

    # the Radex tau does not take hyperfines into consideration, and hence the factor of two output
    return tex, tau



def mock_textau_22(SNR="low"):
    # all the tex and tau are calculated based on Radex on-line (van der Tak)

    if SNR == "xxlow":
        # Tpeak ~0.08 K (below GAS rms noise of 0.15 K)
        # assuming the excitation condition of Tkin = 10K,  n = 1e3 cm^-3, N = 5e13 cm^-2, "sigma" = 0.2
        tex = 3.2
        tau = 0.18

    elif SNR == "xlow":
        # Tpeak ~0.16 K
        # assuming the excitation condition of Tkin = 10K,  n = 2e3 cm^-3, N = 6e13 cm^-2, "sigma" = 0.2
        tex = 3.6
        tau = 0.19

    elif SNR == "low":
        # Tpeak ~0.47 K
        tex = 5.7
        tau = 0.17

    elif SNR == "med":
        # Tpeak ~1.0 K
        # assuming the excitation condition of Tkin = 10K,  n = 1e4 cm^-3, N = 2e14 cm^-2, "sigma" = 0.2
        tex = 5.8
        tau = 0.40

    elif SNR == "high":
        # Tpeak ~5.9 K
        # assuming the excitation condition of Tkin = 15K,  n = 1e4 cm^-3, N = 7e14 cm^-2, "sigma" = 0.2
        tex = 9.0
        tau = 3.1

    else:
        print "[ERROR] the passed in SNR value is not recognized!"
        return None

    return tex, tau
