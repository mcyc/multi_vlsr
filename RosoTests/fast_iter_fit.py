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
reload(ammv)

#-----------------------------------------------------------------------------------------------------------------------

def fit_2comp(cubename, guesses, **kwargs):
    # get the cube we wish to fit
    cube = SpectralCube.read(cubename)

    mean_spec = get_mean_spec(cube, linename=kwargs['linename'])

    spectrum = get_cubespec(cube)

    def iter_fit(spc_cnv, spc, ncomp):
        # a function to fit the convovled spctrum (spc_cnv) first, and use the fitted result to fit the native spectrum
        kwargs['ncomp'] = ncomp
        # fit the mean spectrum first
        sp_cnv = fit_spec(spectrum=spc_cnv.copy(), guesses=guesses, **kwargs)
        gg = sp_cnv.specfit.modelpars
        gg = np.array([gg]).swapaxes(0, 1)
        # use the mean spectrum
        return fit_spec(spectrum=spc.copy(), guesses=gg, **kwargs)

    # perform fits iteratively
    spec_1comp = iter_fit(mean_spec, spectrum, ncomp=1)
    #spec_2comp = iter_fit(mean_spec, spectrum, ncomp=2)

    return spec_1comp

    #spectrum.specfit.modelpars
    #spc.specfit.modelerrs


def get_mean_spec(cube, linename="oneone"):
    # get the mean spectrum of the entire cube
    # note: masking may be desired in case of nan values
    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube)

    spc = cube.mean(axis=(1, 2))
    return prep_spec(spc, linename)



def get_cubespec(cube, refpix=None, linename="oneone"):
    # read a cube file and extract only the spectrum at the reference pixel
    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube)

    if refpix is None:
        # use the central pixel as reference
        shape = cube._data.shape
        refpix = (shape[1]/2, shape[2]/2)

    spc = cube[:, refpix[0], refpix[1]]

    return prep_spec(spc, linename)


def prep_spec(OneDSpectrum, linename="oneone"):
    # take a spectral_cube OneDSpectrum and make it a pyspeckit Spectrum ready to be fitted
    spc = OneDSpectrum
    spectrum = pyspeckit.Spectrum(data=spc.value, xarr=spc.spectral_axis, unit=spc.unit,
                                      xarrkwargs={'unit': spc.spectral_axis.unit})
    if spectrum.xarr.refX is None:
        spectrum.xarr.refX = freq_dict[linename]*u.Hz
    spectrum.xarr.velocity_convention = 'radio'
    spectrum.xarr = spectrum.xarr.as_unit(u.km/u.s)

    return spectrum


def fit_spec(spectrum, guesses, **kwargs):

    ncomp = kwargs['ncomp']

    if not 'linename' in kwargs:
        kwargs['linename'] = "oneone"

    fitter = ammv.nh3_multi_v_model_generator(n_comp=ncomp, linenames=[kwargs['linename']])
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

    # define acceptable v range based on the provided or determined median velocity
    vmax = v_median + v_peak_hwidth
    vmin = v_median - v_peak_hwidth

    # set the fit parameter limits (consistent with GAS DR1)
    #Tbg = 2.8       # K
    Texmin = 3.0    # K; a more reasonable lower limit (5 K T_kin, 1e3 cm^-3 density, 1e13 cm^-2 column, 3km/s sigma)
    Texmax = 40    # K; DR1 T_k for Orion A is < 35 K. T_k = 40 at 1e5 cm^-3, 1e15 cm^-2, and 0.1 km/s yields Tex = 37K
    sigmin = 0.07   # km/s
    sigmax = 2.5    # km/s; for Larson's law, a 10pc cloud has sigma = 2.6 km/s
    taumax = 100.0  # a reasonable upper limit for GAS data. At 10K and 1e5 cm^-3 & 3e15 cm^-2 -> 70
    taumin = 0.2   # note: at 1e3 cm^-3, 1e13 cm^-2, 1 km/s linewidth, 40 K -> 0.15
    eps = 0.001 # a small perturbation that can be used in guesses

    # get the guesses based on moment maps
    # tex and tau guesses are chosen to reflect low density, diffusive gas that are likley to have low SNR
    gg = moment_guesses(np.array([m1]), np.array([m2]), ncomp, sigmin=sigmin, moment0=np.array([m0]))

    if guesses is None:
        guesses = gg

    else:
        # fill in the blanks with moment guesses
        guesses[guesses==0] = np.nan
        gmask = np.isfinite(guesses)
        guesses[~gmask] = gg[~gmask]

        # fill in the failed sigma guesses with moment guesses
        gmask = guesses[1::4] < sigmin
        guesses[1::4][gmask] = gg[1::4][gmask]

        print "user provided guesses accepted"

    # The guesses should be fine in the first case, but just in case, make sure the guesses are confined within the
    # appropriate limits
    guesses[::4][guesses[::4] > vmax] = vmax
    guesses[::4][guesses[::4] < vmin] = vmin
    guesses[1::4][guesses[1::4] > sigmax] = sigmax
    guesses[1::4][guesses[1::4] < sigmin] = sigmin + eps
    guesses[2::4][guesses[2::4] > Texmax] = Texmax
    guesses[2::4][guesses[2::4] < Texmin] = Texmin
    guesses[3::4][guesses[3::4] > taumax] = taumax
    guesses[3::4][guesses[3::4] < taumin] = taumin

    spectrum.specfit(fittype='nh3_multi_v', guesses=guesses,
                     limitedmax=[True,True,True,True]*ncomp,
                     maxpars=[vmax, sigmax, Texmax, taumax]*ncomp,
                     limitedmin=[True,True,True,True]*ncomp,
                     minpars=[vmin, sigmin, Texmin, taumin]*ncomp, **kwargs)

    return spectrum






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

    return fit_spec(spectrum=cnv_spectrum, guesses=guesses, **kwargs)


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


def moment_guesses(moment1, moment2, ncomp, sigmin=0.07, tex_guess=3.2, tau_guess=0.5, moment0=None):
    '''
    Make reasonable guesses for the multiple component fits
    :param moment1:
    :param moment2:
    :param ncomp:
    :param sigmin:
        <float> default at 0.07 km/s, the spectral resolution of the GAS channels
    :param tex_guess:
    :param tau_guess:
    :return:
    '''

    # define max and min values of tex and tau to use for the test
    # a spectrum with tex and tau values both below the specified minima has an intensity below the expected GAS rms
    tex_max = 8.0
    tau_max = 1.0
    tex_min = 3.1
    tau_min = 0.3

    if moment0 is not None:
        print "[WARNING]: moment0 map is provided, thus the user-provided tex and tau will not be used"
        # normalize the moment 0 map with respect to the norm_ref percentile value
        # e.g., 95 percentile value being normalized to have a value of 0.95
        norm_ref = 99.73
        mom0high = np.percentile(moment0[np.isfinite(moment0)], norm_ref)
        print "moment 0 value at {0} percentile: {1}".format(norm_ref, mom0high)
        # may want to modify this normalization to be something a little simpler or physical (i.e., 99.73/100 ~ 1)
        m0Norm = moment0.copy()*norm_ref/100.0/mom0high
        tex_guess = m0Norm*tex_max
        tau_guess = m0Norm*tau_max

    m1 = moment1
    m2 = moment2

    # Guess linewidth (the current recipe works okay, but potential improvements can be made.
    gs_sig = m2/ncomp
    gs_sig[gs_sig < sigmin] = sigmin
    # note 0.08 k is narrow enough to be purely thermal @ ~10 K

    # there are 4 parameters for each v-component
    gg = np.zeros((ncomp*4,)+m1.shape)

    if ncomp == 1:
        gg[0,:] = m1                 # v0 centriod
        gg[1,:] = gs_sig             # v0 width
        gg[2,:] = tex_guess          # v0 T_ex
        gg[3,:] = tau_guess          # v0 tau

    # using a working recipe (assuming a bright and a faint componet)
    if ncomp == 2:
        #sigmaoff = 0.25
        sigmaoff = 0.4
        tau2_frac = 0.25                    # the tau weight of the second component relative to the total fraction
        gg[0,:] = m1 - sigmaoff*m2         # v0 centriod
        gg[1,:] = gs_sig                   # v0 width
        gg[2,:] = tex_guess                # v0 T_ex
        gg[3,:] = tau_guess*(1-tau2_frac)  # v0 tau
        gg[4,:] = m1 + sigmaoff*m2         # v1 centriod
        gg[5,:] = gs_sig                   # v1 width
        gg[6,:] = tex_guess*0.8            # v1 T_ex
        gg[7,:] = tau_guess*tau2_frac      # v1 tau

    # using a generalized receipe that have not been tested (lots of room for improvement!)
    if ncomp > 2:
        for i in range (0, ncomp):
            gg[i,  :] = m1+(-1.0+i*1.0/ncomp)*0.5*m2 # v0 centriod (step through a range fo velocities within sigma_v)
            gg[i+1,:] = gs_sig                   # v0 width
            gg[i+2,:] = tex_guess*0.8            # v0 T_ex
            gg[i+3,:] = tau_guess/ncomp*0.25     # v0 tau

    #print "guesses:"
    #print gg

    # ensure the tex and tau guesses falls within the guessing limits
    tex_guess[tex_guess < tex_min] = tex_min
    tex_guess[tex_guess > tex_max] = tex_max
    tau_guess[tau_guess < tau_min] = tau_min
    tau_guess[tau_guess > tau_max] = tau_max

    return gg


#-----------------------------------------------------------------------------------------------------------------------

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