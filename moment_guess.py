import numpy as np
from astropy.io import fits
from astropy import units as u
from spectral_cube import SpectralCube
from radio_beam import Beam
import pyspeckit
from pyspeckit.spectrum.models.ammonia_constants import freq_dict, voff_lines_dict
from astropy.stats import mad_std

# define max and min values of tex and tau to use for the test
# a spectrum with tex and tau values both below the specified minima has an intensity below the expected GAS rms
tex_max = 8.0
tau_max = 1.0
tex_min = 3.1
tau_min = 0.3


def master_guess(spectrum, ncomp, sigmin = 0.07, v_peak_hwidth=3.0, v_atpeak=None):

    '''
    if guesses is not None:
        v_guess = guesses[::4]
        v_guess[v_guess == 0] = np.nan
        v_median = np.nanmedian(v_guess)
        m0, m1, m2 = main_hf_moments(spectrum, window_hwidth=v_peak_hwidth, v_atpeak=v_median)
    else:
        m0, m1, m2 = main_hf_moments(spectrum, window_hwidth=v_peak_hwidth)
        v_median = np.median(m1[np.isfinite(m1)])
    '''

    m0, m1, m2 = window_moments(spectrum, window_hwidth=v_peak_hwidth, v_atpeak=v_atpeak)

    # estimate the rms level, and pass to the spectrum
    #rms = get_rms(spectrum, window_hwidth=v_peak_hwidth, v_atpeak=v_median)
    #spectrum.error = rms*np.ones_like(spectrum.data)

    # get the guesses based on moment maps
    gg = moment_guesses(np.array([m1]), np.array([m2]), ncomp, sigmin=sigmin, moment0=np.array([m0]))
    return gg


def guess_2comp():
    return gg


def window_moments(spectrum, window_hwidth, v_atpeak=None):
    '''
    find moments within a given window (e.g., around the main hyperfine lines)

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

    if moment0 is not None:
        #print "[WARNING]: moment0 map is provided, thus the user-provided tex and tau will not be used"
        # normalize the moment 0 map with respect to the norm_ref percentile value
        # e.g., 95 percentile value being normalized to have a value of 0.95
        norm_ref = 99.73
        mom0high = np.percentile(moment0[np.isfinite(moment0)], norm_ref)
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


    # ensure the tex and tau guesses falls within the guessing limits
    tex_guess[tex_guess < tex_min] = tex_min
    tex_guess[tex_guess > tex_max] = tex_max
    tau_guess[tau_guess < tau_min] = tau_min
    tau_guess[tau_guess > tau_max] = tau_max

    return gg