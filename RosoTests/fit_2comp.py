import numpy as np
from spectral_cube import SpectralCube
import pyspeckit
import sys, os #, errno, time

import itertools
from itertools import repeat as rp

# add the parent directory to the paths

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import aic
import fast_iter_fit as fifit
from multiprocessing import Pool, cpu_count
import tqdm
import gc
import moment_guess as mmg


########################################################################################################################
# to hide message from pyspeckit

# example
'''
with HiddenPrints():
    print("This will not be printed")
'''

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


########################################################################################################################
# create a multi-processing  wrapper

def run(cubenames, guesses=None, paraname=None, snr_min=3, linename="oneone", rec_wide_vsep=True, n_cpu=None):

    nCubes = len(cubenames)
    print("number of cubes: {}".format(nCubes))

    results = []

    if n_cpu is None:
        n_cpu = cpu_count() - 1

    if n_cpu > 1:
        print "number of cpu used: {}".format(n_cpu)
        # multi-process if more than one cpu is allocated
        pool = Pool(n_cpu)

        for j in tqdm.tqdm(pool.imap(f_star, itertools.izip(cubenames, rp(rec_wide_vsep), rp(guesses), rp(paraname),
                                                            rp(snr_min), rp(linename))), total=nCubes, mininterval=0.01):
            if j is not None:
                results.append(j)
                gc.collect()
                # need something to close the .fits file?

    else:
        print "number of cpu used: {}, no multi-processing is used".format(n_cpu)
        for cubename in tqdm.tqdm(cubenames, mininterval=0.01):
            result = f(cubename, rec_wide_vsep, guesses, paraname, snr_min, linename)
            results.append(result)


    para1, err1, para2, err2, likelyhood, rms = zip(*results)
    return para1, err1, para2, err2, likelyhood, rms


def f(cubename, rec_wide_vsep, guesses, paraname, snr_min, linename):

    results = fit_2comp(cubename, rec_wide_vsep=rec_wide_vsep, guesses=guesses, paraname=paraname, snr_min=snr_min,
                        linename=linename)
    return results


def f_star(paras):
    with HiddenPrints():
        results = f(*paras)
    return results


########################################################################################################################


def fit_2comp(cubename, rec_wide_vsep = True, guesses=None, **kwargs):

    # likihood threshold for model selection
    lnk_thresh = 5

    # get the cube we wish to fit
    cube = SpectralCube.read(cubename)

    # use a mask to mimic convolution to twice the beamsize
    # mean_mask = np.ones((3,3),dtype=bool)
    mean_mask = np.array([[False, False, False],
                     [True,  True, True],
                     [False, True, False]])

    mean_spec = fifit.get_mean_spec(cube, linename=kwargs['linename'], mask=mean_mask)

    spectrum = fifit.get_cubespec(cube)


    def get_residual_spec(spectrum):
        sp_r = pyspeckit.Spectrum(data=spectrum.specfit.fullresiduals.copy(), xarr=spectrum.xarr, header=spectrum.header)
        return sp_r


    def iter_fit(spc_cnv, spc, ncomp, sguesses=None, widewVSep=False, returnCnvRes=False):
        # a function to fit the convovled spctrum (spc_cnv) first, and use the fitted result to fit the native spectrum
        kwargs['ncomp'] = ncomp
        kwargs['widewVSep'] = widewVSep

        if sguesses is None:
            # sguesses superceeds the global guesses if provided
            sguesses = guesses

        # fit the mean spectrum first
        if not spc_cnv is None:
            sp_cnv = fifit.fit_spec(spectrum=spc_cnv.copy(), guesses=sguesses, **kwargs)
            gg = sp_cnv.specfit.modelpars
            gg = np.array([gg]).swapaxes(0, 1)
        elif not sguesses is None:
            gg = sguesses
        else:
            print("[ERROR] both spc_cnv and spc cannot be None at the same time")

        fit_result = fifit.fit_spec(spectrum=spc.copy(), guesses=gg, **kwargs)
        # use the mean spectrum
        if returnCnvRes:
            return fit_result, get_residual_spec(sp_cnv)
        else:
            return fit_result

    # perform fits iteratively
    spec_1comp, sp_r = iter_fit(mean_spec, spectrum, ncomp=1, returnCnvRes=True)
    spec_2comp = iter_fit(mean_spec, spectrum, ncomp=2)

    # mask over were both models are non-zero
    mask1 = spec_1comp.specfit.model > 0
    mask2 = spec_2comp.specfit.model > 0
    mask = np.logical_or(mask1, mask2)

    def get_comp_AICc(spectrum1, spectrum2, p1, p2, mask):
        chi1, N1 = fifit.get_chisq(spectrum1, expand=20, reduced=False, usemask=True, mask=mask)
        chi2, N2 = fifit.get_chisq(spectrum2, expand=20, reduced=False, usemask=True, mask=mask)
        aicc1 = aic.AICc(chi1, p1, N1)
        aicc2 = aic.AICc(chi2, p2, N1)
        lnK = (aicc1 - aicc2) / 2.0
        return lnK

    # calculate the likelihood
    likelyhood = get_comp_AICc(spec_1comp, spec_2comp, p1=4, p2=8, mask=mask)

    def spec_moment_guess(sp_r, window_hwidth=3.0, v_atpeak=None, iter_refine=False):
        # wrapper to make mmg.moment_guesses() competitable with specs
        #moms = mmg.window_moments(sp_r, window_hwidth=window_hwidth, v_atpeak=v_atpeak, iter_refine=iter_refine)
        moms = mmg.window_moments_spc(sp_r, window_hwidth=window_hwidth, v_atpeak=v_atpeak, iter_refine=iter_refine)
        gg = mmg.moment_guesses(np.array([moms[1]]), np.array([moms[2]]), ncomp=1, moment0=np.array([moms[0]]))
        return gg

    if rec_wide_vsep and (likelyhood < lnk_thresh):
        # try to recover second component that may have been missed in the first 2-slab fit attempt
        # this is carried over where one-slab is determined to be a better fit in the first try

        # use 1-slab model parameters as the 1st component guess
        gg1 = spec_1comp.specfit.modelpars
        gg1 = np.array(gg1)[:, np.newaxis]

        # use the 1-slab fit residuals as the 2nd component guess (note, this does not take advantage of the nearby
        # pixels)

        #gg2 = spec_moment_guess(sp_r, window_hwidth=3.0, v_atpeak=gg1[0])
        gg2 = spec_moment_guess(sp_r, window_hwidth=3.5, v_atpeak=gg1[0], iter_refine=False)

        '''
        if not np.all(np.isfinite(gg2)):
            # try again without uusing the moment gueuss for peak
            # note: this shouldn't be needed if mmg.master_guess() is not used
            gg2 = spec_moment_guess(sp_r, window_hwidth=3.0, v_atpeak=None)
        '''

        if np.all(np.isfinite(gg2)):
            # if all the guesses are finite, perform the fit

            # combine the guesses
            sguesses = np.concatenate((gg1, gg2))

            spec_2wcomp = iter_fit(mean_spec, spectrum, ncomp=2, sguesses=sguesses, widewVSep=False)
            #spec_2wcomp = iter_fit(None, spectrum, ncomp=2, sguesses=sguesses, widewVSep=False)
            #spec_2wcomp = iter_fit(mean_spec, spectrum, ncomp=2, widewVSep=True)

            # mask over were both models are non-zero
            mask1 = spec_1comp.specfit.model > 0
            mask2 = spec_2wcomp.specfit.model > 0
            mask = np.logical_or(mask1, mask2)

            likelyhood_w = get_comp_AICc(spec_1comp, spec_2wcomp, p1=4, p2=8, mask=mask)

            if likelyhood_w > lnk_thresh:
                likelyhood = likelyhood_w
                spec_2comp = spec_2wcomp

    # calculate the rms at where the better fitted model is non-zero
    if likelyhood > lnk_thresh:
        rms = fifit.get_rms(spec_2comp, expand=10, usemask=True, mask=mask2)
    else:
        rms = fifit.get_rms(spec_1comp, expand=10, usemask=True, mask=mask1)


    # returning objects in  multiprocessing is tricky, thus we are only returing fitted values here
    para1 = spec_1comp.specfit.modelpars
    err1 = spec_1comp.specfit.modelerrs
    para2 = spec_2comp.specfit.modelpars
    err2 = spec_2comp.specfit.modelerrs
    return para1, err1, para2, err2, likelyhood, rms