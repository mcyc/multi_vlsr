import numpy as np
from astropy.io import fits
from astropy import units as u
from spectral_cube import SpectralCube
from radio_beam import Beam
import pyspeckit
from pyspeckit.spectrum.models.ammonia_constants import freq_dict, voff_lines_dict
from astropy.stats import mad_std
import sys, os #, errno, time
# add the parent directory to the paths
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import multi_v_fit as mvf
import iterative_fit as itf
import ammonia_multiv as ammv
import aic
import warnings
import moment_guess as momgue
reload(mvf)


#-----------------------------------------------------------------------------------------------------------------------

def fit_2comp(cubename, guesses, **kwargs):
    # get the cube we wish to fit
    cube = SpectralCube.read(cubename)

    # bing 4 pixels into a spectrum to emulate a cube that is convolved to twice the resolution
    mask = np.array([[False, False, False],
                     [True,  True, True],
                     [False, True, False]])

    mean_spec = get_mean_spec(cube, linename=kwargs['linename'], mask=mask)

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
    spec_2comp = iter_fit(mean_spec, spectrum, ncomp=2)

    def get_comp_AICc(spectrum1, spectrum2, p1, p2):
        model1 = spectrum1.specfit.model
        model2 = spectrum2.specfit.model
        mask1 = model1 > 0
        mask2 = model2 > 0
        mask = np.logical_or(mask1, mask2)
        chi1, N1 = get_chisq(spectrum1, expand=20, reduced=False, usemask=True, mask=mask)
        chi2, N2 = get_chisq(spectrum2, expand=20, reduced=False, usemask=True, mask=mask)
        aicc1 = aic.AICc(chi1, p1, N1)
        aicc2 = aic.AICc(chi2, p2, N1)
        likelyhood = (aicc1 - aicc2) / 2.0
        return likelyhood

    likelyhood = get_comp_AICc(spec_1comp, spec_2comp, p1=4, p2=8)
    print "likelyhood: {}".format(likelyhood)

    return spec_1comp, spec_2comp, likelyhood

    #spectrum.specfit.modelpars
    #spc.specfit.modelerrs


def get_mean_spec(cube, linename="oneone", mask=None):
    # get the mean spectrum of the entire cube
    # note: masking may be desired in case of nan values
    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube)

    if mask is not None:
        if mask.ndim == 2:
            mask = np.broadcast_to(mask, cube.shape)
        spc = (cube.with_mask(mask)).mean(axis=(1, 2))
    else:
        spc = cube.mean(axis=(1, 2))

    return prep_spec(spc, cube.header, linename)


def get_cubespec(cube, refpix=None, linename="oneone"):
    # read a cube file and extract only the spectrum at the reference pixel
    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube)

    if refpix is None:
        # use the central pixel as reference
        shape = cube._data.shape
        refpix = (shape[1]/2, shape[2]/2)

    spc = cube[:, refpix[0], refpix[1]]
    #print cube.header
    return prep_spec(spc, cube.header, linename)


def prep_spec(OneDSpectrum, header, linename="oneone"):
    # take a spectral_cube OneDSpectrum and make it a pyspeckit Spectrum ready to be fitted
    spc = OneDSpectrum
    spectrum = pyspeckit.Spectrum(data=spc.value, xarr=spc.spectral_axis,
                                      xarrkwargs={'unit': spc.spectral_axis.unit}, header=header)
    if spectrum.xarr.refX is None:
        if ('CUNIT3' in header) and ('RESTFRQ' in header):
            spectrum.xarr.refX = header['RESTFRQ'] * u.Unit(header['CUNIT3'])
        else:
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

    # estimate the rms level, and pass to the spectrum
    #rms = get_rms(spectrum, window_hwidth=v_peak_hwidth, v_atpeak=v_median)
    #spectrum.error = rms*np.ones_like(spectrum.data)

    # set the fit parameter limits (consistent with GAS DR1)
    #Tbg = 2.8       # K
    Texmin = 3.0    # K; a more reasonable lower limit (5 K T_kin, 1e3 cm^-3 density, 1e13 cm^-2 column, 3km/s sigma)
    Texmax = 40    # K; DR1 T_k for Orion A is < 35 K. T_k = 40 at 1e5 cm^-3, 1e15 cm^-2, and 0.1 km/s yields Tex = 37K
    sigmin = 0.07   # km/s
    sigmax = 2.5    # km/s; for Larson's law, a 10pc cloud has sigma = 2.6 km/s
    taumax = 100.0  # a reasonable upper lim it for GAS data. At 10K and 1e5 cm^-3 & 3e15 cm^-2 -> 70
    taumin = 0.2   # note: at 1e3 cm^-3, 1e13 cm^-2, 1 km/s linewidth, 40 K -> 0.15
    eps = 0.001 # a small perturbation that can be used in guesses

    # get guesses based on moment maps
    v_atpeak=None
    v_median=None

    if guesses is not None:
        v_guess = guesses[::4]
        v_guess[v_guess == 0] = np.nan
        v_median = np.nanmedian(v_guess)
        v_atpeak = v_median

    gg = momgue.master_guess(spectrum, ncomp, sigmin=sigmin, v_peak_hwidth=v_peak_hwidth, v_atpeak=v_atpeak)

    # define acceptable v range based on the provided or determined median velocity
    if v_median is None:
        # use the median value of the moment vlsr guesses
        v_median = np.nanmedian(np.array(gg[::4]))

    vmax = v_median + v_peak_hwidth
    vmin = v_median - v_peak_hwidth

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

        #print "user provided guesses accepted"

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

def main_hf_moments_old(spectrum, window_hwidth, v_atpeak=None):
    return momgue.window_moments(spectrum, window_hwidth, v_atpeak=v_atpeak)


def get_rms_prefit(spectrum, window_hwidth, v_atpeak):

    s = spectrum

    #vsys = ThisRegion['VAVG'] * u.km / u.s
    vsys = v_atpeak*u.km/u.s
    #throw = 2 * u.km / u.s + ThisRegion['VRANGE'] * u.km / u.s / 2
    throw = window_hwidth*u.km/u.s
    voff11 = voff_lines_dict['oneone']

    mask = np.ones(s.shape[0], dtype=np.bool)

    for deltav in voff11:
        mask *= (np.abs(s.xarr - (deltav * u.km / u.s + vsys)) > throw)

    d_rms = s.data.copy()


    return mad_std(d_rms[mask])



def get_chisq(spectrum, expand=20, reduced=True, usemask=True, mask=None):
    '''
    cube : SpectralCube

    model: numpy array

    expand : int
        Expands the region where the residual is evaluated by this many channels in the spectral dimension

    reduced : boolean
        Whether or not to return the reduced chi-squared value or not

    usemask: boolean
        Whether or not to mask out some parts of the data.
        If no mask is provided, it masks out samples with model values of zero.

    mask: boolean array
        A mask stating which array elements the chi-squared values are calculated from
    '''

    import scipy.ndimage as nd

    model = spectrum.specfit.model

    if usemask:
        if mask is None:
            mask = model > 0
    else:
        mask = ~np.isnan(model)

    residual = spectrum.specfit.residuals

    # This calculates chisq over the region where the fit is non-zero
    # plus a buffer of size set by the expand keyword.

    selem = np.ones(expand, dtype=np.bool)
    #selem.shape += (1, 1,)
    mask = nd.binary_dilation(mask, selem)
    mask = mask.astype(np.float)
    chisq = np.sum((residual * mask) ** 2, axis=0)

    if reduced:
        chisq /= np.sum(mask, axis=0)

    # This produces a robust estimate of the RMS along every line of sight:
    # (alternatively, we can use mad_std from astropy?)
    diff = residual - np.roll(residual, 2, axis=0)
    rms = 1.4826 * np.nanmedian(np.abs(diff), axis=0) / 2 ** 0.5

    chisq /= rms ** 2

    if reduced:
        # return the reduce chi-squares values
        return chisq

    else:
        # return the ch-squared values and the number of data points used
        return chisq, np.sum(mask, axis=0)


def get_rms(spectrum, expand=20, usemask=True, mask=None):
    '''
    return rms over where no model is fitted
    '''

    import scipy.ndimage as nd

    model = spectrum.specfit.model

    if usemask:
        if mask is None:
            mask = model > 0
    else:
        mask = ~np.isnan(model)

    residual = spectrum.specfit.residuals

    # Mask over the region where the fit is non-zero plus a buffer of size set by the expand keyword.
    selem = np.ones(expand, dtype=np.bool)
    #selem.shape += (1, 1,)
    mask = nd.binary_dilation(mask, selem)
    #mask = mask.astype(np.float)

    # Now get where the emission is zero and estimate the rms
    # This produces a robust estimate of the RMS along every line of sight:
    diff = residual - np.roll(residual, 2, axis=0)

    if len(diff) - mask.sum() > 10:
        # only use the mask if there are more than 10 model-free chanels
        diff = diff[~mask]

    rms = 1.4826 * np.nanmedian(np.abs(diff), axis=0) / 2 ** 0.5
    #print "rms: {}; \t sample size: {}".format(rms, len(diff))

    return rms



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