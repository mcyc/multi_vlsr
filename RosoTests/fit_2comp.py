import numpy as np
from spectral_cube import SpectralCube
import pyspeckit
import sys, os #, errno, time
# add the parent directory to the paths
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import aic
import fast_iter_fit as fifit
reload(fifit)
from multiprocessing import Pool, cpu_count
import tqdm
import gc

def run(cubenames, guesses_pp, kwargs_pp, ncpu=None):
    global guesses, kwargs
    guesses = guesses_pp
    kwargs = kwargs_pp

    # use a mask to mimic convolution to twice the beamsize
    global mean_mask

    mean_mask = np.array([[False, False, False],
                     [True,  True, True],
                     [False, True, False]])

    #mean_mask = np.ones((3,3),dtype=bool)

    results = []

    if ncpu is None:
        ncpu = cpu_count() - 1
        print "number of cpu used: {}".format(ncpu)

    elif ncpu == 1:
        # single processing
        print "number of cpu specified is {}, no multi-processing is used".format(ncpu)
        #for cubename in cubenames:

        for cubename in tqdm.tqdm(cubenames, total=len(cubenames), mininterval=0.01):
            results.append(fit_2comp(cubename))
            gc.collect()
        para1, err1, para2, err2, likelyhood, rms = zip(*results)
        return para1, err1, para2, err2, likelyhood, rms


    pool = Pool(ncpu)  # Create a multiprocessing Pool

    #results = []
    print "cube length: {}".format(len(cubenames))
    for i in tqdm.tqdm(pool.imap(fit_2comp, cubenames), total=len(cubenames), mininterval=0.01):
        results.append(i)
        gc.collect()

    para1, err1, para2, err2, likelyhood, rms = zip(*results)
    #para1, err1, para2, err2, likelyhood = zip(*(tqdm.tqdm(pool.imap(fit_2comp, cubenames), total=len(cubenames))))

    return para1, err1, para2, err2, likelyhood, rms



def fit_2comp(cubename, rec_wide_vsep = True):

    # likihood threshold for model selection
    lnk_thresh = 5

    # get the cube we wish to fit
    cube = SpectralCube.read(cubename)

    mean_spec = fifit.get_mean_spec(cube, linename=kwargs['linename'], mask=mean_mask)

    spectrum = fifit.get_cubespec(cube)

    def iter_fit(spc_cnv, spc, ncomp, sguesses=None, widewVSep=False):
        # a function to fit the convovled spctrum (spc_cnv) first, and use the fitted result to fit the native spectrum
        kwargs['ncomp'] = ncomp
        kwargs['widewVSep'] = widewVSep

        if sguesses is None:
            # sguesses superceeds the global guesses if provided
            sguesses = guesses

        # fit the mean spectrum first
        sp_cnv = fifit.fit_spec(spectrum=spc_cnv.copy(), guesses=sguesses, **kwargs)
        gg = sp_cnv.specfit.modelpars
        gg = np.array([gg]).swapaxes(0, 1)
        # use the mean spectrum
        return fifit.fit_spec(spectrum=spc.copy(), guesses=gg, **kwargs)

    # perform fits iteratively
    spec_1comp = iter_fit(mean_spec, spectrum, ncomp=1)
    spec_2comp = iter_fit(mean_spec, spectrum, ncomp=2)

    # mask over were both models are non-zero
    mask1 = spec_1comp.specfit.model > 0
    mask2 = spec_2comp.specfit.model > 0
    mask = np.logical_or(mask1, mask2)

    def get_comp_AICc(spectrum1, spectrum2, p1, p2, mask):
        mask = np.logical_or(mask1, mask2)
        chi1, N1 = fifit.get_chisq(spectrum1, expand=20, reduced=False, usemask=True, mask=mask)
        chi2, N2 = fifit.get_chisq(spectrum2, expand=20, reduced=False, usemask=True, mask=mask)
        aicc1 = aic.AICc(chi1, p1, N1)
        aicc2 = aic.AICc(chi2, p2, N1)
        likelyhood = (aicc1 - aicc2) / 2.0
        return likelyhood

    # calculate the likelihood
    likelyhood = get_comp_AICc(spec_1comp, spec_2comp, p1=4, p2=8, mask=mask)

    if rec_wide_vsep and (likelyhood < lnk_thresh):
        # try to recover second component that may have been missed in the first 2-slab fit attempt
        # this is carried over where one-slab is determined to be a better fit in the first try

        def get_residual_spec(spectrum):
            sp_r = spectrum.copy()
            sp_r.data = spectrum.specfit.fullresiduals
            return sp_r

        import moment_guess as mmg

        # use the 1-slab fit residuals as the 2nd component guess
        sp_r = get_residual_spec(spec_1comp)
        gg2 = mmg.master_guess(sp_r, ncomp=1, snr_cut=2)

        if np.all(np.isfinite(gg2)):
            # if all the guesses are finite, perform the fit

            # use 1-slab model parameters as the 1st component guess
            gg1 = spec_1comp.specfit.modelpars
            gg1 = np.array(gg1)[:, np.newaxis]

            # combine the guesses
            sguesses = np.concatenate((gg1, gg2))

            spec_2wcomp = iter_fit(mean_spec, spectrum, ncomp=2, sguesses=sguesses, widewVSep=True)
            #spec_2wcomp = iter_fit(mean_spec, spectrum, ncomp=2, widewVSep=True)
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