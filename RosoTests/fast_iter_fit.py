import numpy as np
from astropy.io import fits
from astropy import units as u
from spectral_cube import SpectralCube
from radio_beam import Beam
import pyspeckit
from pyspeckit.spectrum.models.ammonia_constants import freq_dict, voff_lines_dict
import sys, os #, errno, time
# add the parent directory to the paths
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import multi_v_fit as mvf
import iterative_fit as itf
import ammonia_multiv as ammv

#-----------------------------------------------------------------------------------------------------------------------

def fit_2comp():
    fit_spec()


def get_cubespec(cubename, refpix=None, linename="oneone"):
    # read a cube file and extract only the spectrum at the reference pixel

    cube = SpectralCube.read(cubename)

    if refpix is None:
        # use the central pixel as reference
        shape = cube._data.shape
        refpix = (shape[1]/2, shape[2]/2)

    spc = cube[:, refpix[0], refpix[1]]

    spectrum = pyspeckit.Spectrum(data=spc.value, xarr=spc.spectral_axis, unit=spc.unit,
                                      xarrkwargs={'unit': spc.spectral_axis.unit})

    if spectrum.xarr.refX is None:
        spectrum.xarr.refX = freq_dict[linename]*u.Hz
    spectrum.xarr.velocity_convention = 'radio'
    spectrum.xarr = spectrum.xarr.as_unit(u.km/u.s)

    return spectrum



def fit_spec(spectrum, ncomp, guesses, linename="oneone"):

    fitter = ammv.nh3_multi_v_model_generator(n_comp = ncomp, linenames=[linename])
    #pcube.specfit.Registry.add_fitter('nh3_multi_v', fitter, fitter.npars)
    spectrum.specfit.Registry.add_fitter('nh3_multi_v', fitter, fitter.npars)

    v_peak_hwidth = 3.0 # km/s (should be sufficient for GAS Orion, but may not be enough for KEYSTONE)

    if guesses is not None:
        v_guess = guesses[::4]
        v_guess[v_guess == 0] = np.nan
        v_median = np.nanmedian(v_guess)
        print "The median of the user provided velocities is: {0}".format(v_median)
        m0, m1, m2 = main_hf_moments(spectrum, window_hwidth=v_peak_hwidth, v_atpeak=v_median)
    else:
        m0, m1, m2 = main_hf_moments(spectrum, window_hwidth=v_peak_hwidth)
        v_median = np.median(m1[np.isfinite(m1)])
        print "median velocity: {0}".format(v_median)

    # set the fit parameter limits (consistent with GAS DR1)
    #Tbg = 2.8       # K
    Texmin = 3.0    # K; a more reasonable lower limit (5 K T_kin, 1e3 cm^-3 density, 1e13 cm^-2 column, 3km/s sigma)
    Texmax = 40    # K; DR1 T_k for Orion A is < 35 K. T_k = 40 at 1e5 cm^-3, 1e15 cm^-2, and 0.1 km/s yields Tex = 37K
    sigmin = 0.07   # km/s
    sigmax = 2.5    # km/s; for Larson's law, a 10pc cloud has sigma = 2.6 km/s
    taumax = 100.0  # a reasonable upper limit for GAS data. At 10K and 1e5 cm^-3 & 3e15 cm^-2 -> 70
    taumin = 0.2   # note: at 1e3 cm^-3, 1e13 cm^-2, 1 km/s linewidth, 40 K -> 0.15
    eps = 0.001 # a small perturbation that can be used in guesses

    return m1






def cubefit(cubename, downsampfactor=2, refpix=None, guesses=None, **kwargs):
    '''
    cubefit(cube11name, ncomp=2, paraname=None, modname=None, chisqname=None, errmap11name=None,
                multicore=1, mask_function=None, snr_min=3.0, linename="oneone", momedgetrim=True, saveguess=False):
    '''

    '''
    root = "conv{0}Xbeam".format(int(np.rint(downsampfactor)))
    cnv_cubename = "{0}_{1}.fits".format(os.path.splitext(cubename)[0], root)

    print "convolve the cube by a factor of: {0}".format(downsampfactor)
    cnv_cube = convolve_sky(cubename, factor=downsampfactor)
    '''

    cnv_spectrum = get_cubespec(cubename)

    return fit_spec(spectrum=cnv_spectrum, ncomp=kwargs['ncomp'], guesses=guesses, linename=kwargs['linename'])

    kwargs_cnv = kwargs.copy()
    kwargs_cnv['paraname'] = None
    kwargs_cnv['momedgetrim'] = False

    # fit the convolved cube to serve as parameter guesses for the full resolution fitting
    cnv_pcube = mvf.cubefit_gen(cnv_cubename, **kwargs_cnv)

    return cnv_pcube

    #data_cnv, hdr_cnv = fits.getdata(kwargs_cnv['paraname'], header=True)
    npara = 4
    ncomp = int(cnv_pcube.parcube.shape[0]/npara)/2



    # the target header for the regridding
    cube_hdr = fits.getheader(cubename)
    hdr_final = get_celestial_hdr(cube_hdr)

    kwargs['guesses'] = guess_from_cnvpara(data_cnv, hdr_cnv, hdr_final, downsampfactor=2)

    pcube = mvf.cubefit_gen(cubename, **kwargs)

    # write the fitted parameters into a fits file
    mvf.save_pcube(pcube, kwargs['paraname'], ncomp=ncomp)

    return pcube


#-----------------------------------------------------------------------------------------------------------------------

def main_hf_moments(spectrum, window_hwidth, v_atpeak=None):
    '''
    find moments for the main hyperfine lines

    :param spectrum:
        <pyspeckit.spectrum.classes.Spectrum>
        the spectrum to take the momentw of

    :param window_hwidth: float
        half-width of the window (in km/s) to be used to isolate the main hyperfine lines from the rest of the spectrum

    -------
    :return: m0
    :return: m1
    :return: m2
    '''

    if v_atpeak is None:
        # use the whole moment map to estimate the speak of the spectrum
        moments = spectrum.moments(unit=u.km/u.s)
        v_atpeak = moments[2]

    vmax = v_atpeak + window_hwidth
    vmin = v_atpeak - window_hwidth

    # Extract the spectrum within the window defined around the main hyperfine components and take moments
    slice = spectrum.slice(vmin, vmax, unit=u.km/u.s)
    moments = slice.moments(unit=u.km/u.s)

    return moments[1], moments[2], moments[3]


def guess_from_cnvpara(data_cnv, header_cnv, header_target):
    # a wrapper to make guesses based on the parameters fitted to the convolved data
    npara = 4
    ncomp = int(data_cnv.shape[0]/npara)/2

    data_cnv = data_cnv.copy()
    # clean up the maps based on vlsr errors
    data_cnv = simple_para_clean(data_cnv, ncomp, npara=npara)
    hdr_conv = get_celestial_hdr(header_cnv)
    data_cnv[data_cnv == 0] = np.nan
    data_cnv = data_cnv[0:npara*ncomp]


    def tautex_renorm(taumap, texmap, tau_thresh = 0.3, tex_thresh = 10.0):

        # attempt to re-normalize the tau & text values at the optically thin regime (where the two are degenerate)
        isthin = np.logical_and(taumap < tau_thresh, np.isfinite(taumap))
        texmap[isthin] = texmap[isthin]*taumap[isthin]/tau_thresh
        taumap[isthin] = tau_thresh

        # optically thin gas are also unlikely to have high spatial density and thus high Tex
        tex_thin = 3.5      # note: at Tk = 30K, n = 1e3, N = 1e13, & sig = 0.2 km.s --> Tex = 3.49 K, tau = 0.8
        hightex = np.logical_and(texmap > tex_thresh, np.isfinite(texmap))
        texmap[hightex] = tex_thin
        taumap[hightex] = texmap[hightex]*taumap[hightex]/tex_thin

        # note, tau values that are too low will be taken care of by refine_each_comp()

        return taumap, texmap

    def refine_each_comp(guess_comp, mask=None):
        # refine guesses for each component, with values outside ranges specified below removed

        Tex_min = 3.0
        Tex_max = 8.0
        Tau_min = 0.2
        Tau_max = 8.0

        disksize = 1.0

        if mask is None:
            mask = master_mask(guess_comp)

        guess_comp[0] = refine_guess(guess_comp[0], min=None, max=None, mask=mask, disksize=disksize)
        guess_comp[1] = refine_guess(guess_comp[1], min=None, max=None, mask=mask, disksize=disksize)

        # re-normalize the degenerated tau & text for the purpose of estimate guesses
        guess_comp[3], guess_comp[2] = tautex_renorm(guess_comp[3], guess_comp[2], tau_thresh = 0.1)

        # place a more "strict" limits for Tex and Tau guessing than the fitting itself
        guess_comp[2] = refine_guess(guess_comp[2], min=Tex_min, max=Tex_max, mask=mask, disksize=disksize)
        guess_comp[3] = refine_guess(guess_comp[3], min=Tau_min, max=Tau_max, mask=mask, disksize=disksize)
        return guess_comp

    for i in range (0, ncomp):
        #data_cnv[i*npara:i*npara+npara] = refine_each_comp(data_cnv[i*npara:i*npara+npara], mmask)
        data_cnv[i*npara:i*npara+npara] = refine_each_comp(data_cnv[i*npara:i*npara+npara])

    # regrid the guess back to that of the original data
    hdr_final = get_celestial_hdr(header_target)

    guesses_final = []

    # regrid the guesses
    for gss in data_cnv:

        newmask = np.isfinite(gss)
        # removal holes with areas that smaller than a 5 by 5 square
        newmask = remove_small_holes(newmask, 25)
        # create a mask to regrid over
        newmask = regrid(newmask, hdr_conv, hdr_final, dmask=None, method='nearest')
        newmask = newmask.astype('bool')
        #newmask = dilation(newmask, disk(2))
        guesses_final.append(regrid(gss, hdr_conv, hdr_final, dmask=newmask))

    return np.array(guesses_final)


def refine_guess(map, min=None, max=None, mask=None, disksize=1):
    # refine parameter maps by outlier-fitering, masking, and interpolating
    map = map.copy()

    if min is not None:
        #map[map<min] = np.nan
        map[map<min] = min
    if max is not None:
        #map[map>max] = np.nan
        map[map>max] = max

    #map = median_filter(map, footprint=disk(disksize))
    return map


def convolve_sky(cube, factor):

    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube)

    hdr = cube.header

    #sanity check
    if hdr['CUNIT1'] != hdr['CUNIT2']:
        print "[ERROR]: the axis units for the do not match each other!"
        return None

    beamunit = getattr(u, hdr['CUNIT1'])
    bmaj = hdr['BMAJ']*beamunit*factor
    bmin = hdr['BMIN']*beamunit*factor
    pa = hdr['BPA']

    beam = Beam(major=bmaj, minor=bmin, pa=pa)

    # convolve
    cnv_cube = cube.convolve_to(beam)

    if cnv_cube.fill_value is not np.nan:
        cnv_cube = cnv_cube.with_fill_value(np.nan)

    return cnv_cube