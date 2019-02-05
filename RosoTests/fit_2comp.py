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

    if ncpu is not None:
        ncpu = cpu_count() - 1

    pool = Pool(ncpu)  # Create a multiprocessing Pool

    '''
    for i, _ in tqdm(enumerate(p.imap_unordered(_foo, range(0, max_)))):
        pbar.update()
    '''

    results = []
    for i in tqdm.tqdm(pool.imap(fit_2comp, cubenames), total=len(cubenames), mininterval=0.01):
        results.append(i)
        gc.collect()

    para1, err1, para2, err2, likelyhood = zip(*results)
    #para1, err1, para2, err2, likelyhood = zip(*(tqdm.tqdm(pool.imap(fit_2comp, cubenames), total=len(cubenames))))

    return para1, err1, para2, err2, likelyhood


def fit_2comp(cubename):
    # get the cube we wish to fit
    cube = SpectralCube.read(cubename)

    mean_spec = fifit.get_mean_spec(cube, linename=kwargs['linename'])

    spectrum = fifit.get_cubespec(cube)

    def iter_fit(spc_cnv, spc, ncomp):
        # a function to fit the convovled spctrum (spc_cnv) first, and use the fitted result to fit the native spectrum
        kwargs['ncomp'] = ncomp
        # fit the mean spectrum first
        sp_cnv = fifit.fit_spec(spectrum=spc_cnv.copy(), guesses=guesses, **kwargs)
        gg = sp_cnv.specfit.modelpars
        gg = np.array([gg]).swapaxes(0, 1)
        # use the mean spectrum
        return fifit.fit_spec(spectrum=spc.copy(), guesses=gg, **kwargs)

    # perform fits iteratively
    spec_1comp = iter_fit(mean_spec, spectrum, ncomp=1)
    spec_2comp = iter_fit(mean_spec, spectrum, ncomp=2)

    def get_comp_AICc(spectrum1, spectrum2, p1, p2):
        model1 = spectrum1.specfit.model
        model2 = spectrum2.specfit.model
        mask1 = model1 > 0
        mask2 = model2 > 0
        mask = np.logical_or(mask1, mask2)
        chi1, N1 = fifit.get_chisq(spectrum1, expand=20, reduced=False, usemask=True, mask=mask)
        chi2, N2 = fifit.get_chisq(spectrum2, expand=20, reduced=False, usemask=True, mask=mask)
        aicc1 = aic.AICc(chi1, p1, N1)
        aicc2 = aic.AICc(chi2, p2, N1)
        likelyhood = (aicc1 - aicc2) / 2.0
        return likelyhood

    likelyhood = get_comp_AICc(spec_1comp, spec_2comp, p1=4, p2=8)

    # returning objects in  multiprocessing is tricky, thus we are only returing fitted values here
    para1 = spec_1comp.specfit.modelpars
    err1 = spec_1comp.specfit.modelerrs
    para2 = spec_2comp.specfit.modelpars
    err2 = spec_2comp.specfit.modelerrs
    return para1, err1, para2, err2, likelyhood