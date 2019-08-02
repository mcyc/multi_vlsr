__author__ = 'mcychen'

#=======================================================================================================================
import os
import numpy as np
from spectral_cube import SpectralCube
from astropy import units as u

import UltraCube as UCube

#=======================================================================================================================

class Region(object):

    def __init__(self, cubePath, paraNameRoot, paraDir=None, cnv_factor=2):

        self.cubePath = cubePath
        self.paraNameRoot = paraNameRoot

        self.ucube = UCube.UCubePlus(cubePath, paraNameRoot=paraNameRoot, paraDir=paraDir, cnv_factor=cnv_factor)

        # for convolving cube
        self.cnv_factor = cnv_factor


    def get_convolved_cube(self, update=True, cnv_cubePath=None, edgetrim_width=None, paraNameRoot=None, paraDir=None):
        get_convolved_cube(self, update=update, cnv_cubePath=cnv_cubePath, edgetrim_width=edgetrim_width,
                           paraNameRoot=paraNameRoot, paraDir=paraDir)


    def get_convolved_fits(self, ncomp, **kwargs):
        get_convolved_fits(self, ncomp, **kwargs)


    def get_fits(self, ncomp, **kwargs):
        get_fits(self, ncomp, **kwargs)



#=======================================================================================================================


def get_convolved_cube(reg, update=True, cnv_cubePath=None, edgetrim_width=None, paraNameRoot=None, paraDir=None):

    if cnv_cubePath is None:
        root = "conv{0}Xbeam".format(int(np.rint(reg.cnv_factor)))
        reg.cnv_cubePath = "{0}_{1}.fits".format(os.path.splitext(reg.cubePath)[0], root)
    else:
        reg.cnv_cubePath = cnv_cubePath

    reg.cnv_para_paths ={}

    if update or (not os.path.isfile(reg.cnv_cubePath)):
        reg.convolve_cube(factor=reg.cnv_factor, savename=cnv_cubePath, edgetrim_width=edgetrim_width)

    if paraNameRoot is None:
        paraNameRoot = "{}_conv{}Xbeam".format(reg.paraNameRoot, int(np.rint(reg.cnv_factor)))

    reg.ucube_cnv = UCube.UCubePlus(cubefile=reg.cnv_cubePath, paraNameRoot=paraNameRoot,
                                     paraDir=paraDir, cnv_factor=reg.cnv_factor)

    # MC: a mechanism is needed to make sure the convolved cube has the same resolution has the cnv_factor


def get_convolved_fits(reg, ncomp, **kwargs):
    if not hasattr(reg, 'ucube_cnv'):
        if 'update' in kwargs:
            reg.get_convolved_cube(update=kwargs['update'])
    reg.ucube_cnv.get_model_fit(ncomp, **kwargs)


def get_fits(reg, ncomp, **kwargs):
    reg.ucube.get_model_fit(ncomp, **kwargs)


def iter_2comp_fit(reg, snr_min=3):
    # force this to be a two-component fit
    ncomp = [1,2]
    import guess_refine as gss_rf

    # convolve the cube and fit it
    get_convolved_cube(reg, update=False)
    get_convolved_fits(reg, ncomp, update=False, snr_min=snr_min)

    # use the result from the convolved cube as guesses for the full resolution fits
    for nc in ncomp:
        pcube = reg.ucube.pcubes[str(nc)]
        para_cnv = np.append(pcube.parcube, pcube.errcube, axis=0)
        if nc == 2:
            para_cnv = gss_rf.quick_2comp_sort(para_cnv, filtsize=3)
        guesses= gss_rf.guess_from_cnvpara(para_cnv, reg.ucube.cube.header, reg.ucube.cube.header)
        reg.ucube.get_model_fit([nc], update=True, guesses=guesses)

def refit_2comp_wide(reg, snr_min=3):
    best_rms = UCube.get_rms(best_res)



def get_best_2comp_residual(reg):
    # get the log-likelihood between the fits
    #reg.ucube.likelihoods['21'] = reg.ucube.get_AICc_likelihood(2, 1)
    lnk21 = reg.ucube.get_AICc_likelihood(2, 1)

    mod1 = reg.ucube.pcubes['1'].get_modelcube()
    mod2 = reg.ucube.pcubes['2'].get_modelcube()

    # get the best model based on the calculated likelihood
    #mask = reg.likelihoods['21'] > 5
    mask = lnk21 > 5
    modbest = mod1.copy()
    modbest[:, mask] = mod2[:, mask]

    # get the residual of the best fit
    modbest[np.isnan(modbest)] = 0
    best_res = reg.ucube.cube._data - modbest
    #best_rms = UCube.get_rms(best_res)

    res_cube = SpectralCube(best_res, reg.ucube.cube.wcs)
    res_cube = res_cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
    return res_cube






def replace_para(pcube, pcube_ref, mask):
    # replace values in masked pixels with the reference values
    pcube_ref = pcube_ref.copy('deep')
    pcube.parcube[:,mask] = pcube_ref.parcube[:,mask]
    pcube.errcube[:,mask] = pcube_ref.errcube[:,mask]

#=======================================================================================================================

def test():
    dirName = "/Users/mcychen/Desktop/UCube_Test/"
    cubepath = "{}/orionA_test_cube.fits".format(dirName)
    fit_2comp(cubepath, ncomp=2, edgetrim_width=5)

