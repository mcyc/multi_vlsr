__author__ = 'mcychen'

#=======================================================================================================================
import os
import numpy as np
from spectral_cube import SpectralCube
from astropy import units as u
from skimage.morphology import dilation

import UltraCube as UCube
import moment_guess as mmg
import convolve_tools as cnvtool

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


def refit_2comp_wide(reg, window_hwidth=4.0, snr_min=3, res_snr_cut=5):
    return None



def get_2comp_wide_guesses(reg, **kwargs):
    return None



def fit_best_2comp_residual_cnv(reg, window_hwidth=4.0, **kwargs):

    # need a mechanism to make sure reg.ucube.pcubes['1'], reg.ucube.pcubes['2'] exists

    cube_res_cnv = get_best_2comp_residual_cnv(reg, masked=True, window_hwidth=window_hwidth, **kwargs)

    ncomp = 1
    moms_res_cnv = mmg.window_moments(cube_res_cnv, window_hwidth=window_hwidth,
                                      v_atpeak=np.nanmedian(reg.ucube.pcubes['1'].parcube[0]))

    gg = mmg.moment_guesses(moms_res_cnv[1], moms_res_cnv[2], ncomp, moment0=moms_res_cnv[0])

    reg_cnv_rez = UCube.UltraCube(cube=cube_res_cnv)
    reg_cnv_rez.fit_cube(ncomp=[1], snr_min=3, guesses=gg)

    return reg_cnv_rez


def get_best_2comp_residual_cnv(reg, masked=True, window_hwidth=4.0, res_snr_cut=5):
    # return convolved residual cube with 'excessive' residual above a peak SNR value of res_snr_cut masked

    # need a mechanism to make sure reg.ucube.pcubes['1'], reg.ucube.pcubes['2'] exists

    res_cube = get_best_2comp_residual(reg)
    best_res = res_cube._data
    cube_res = SpectralCube(data=best_res, wcs=reg.ucube.pcubes['2'].wcs.copy(),
                            header=reg.ucube.pcubes['2'].header.copy())

    if masked:
        #best_rms = UCube.get_rms(best_res)
        best_rms = UCube.get_rms(res_cube._data)

        # calculate the peak SNR value of the best-fit residual over the main hyperfine components
        # (note, peak SNR over ALL the hyperfine may pickup residuals due to hyperfine anomolies)
        res_main_hf = mmg.get_window_slab(res_cube, window_hwidth=window_hwidth,
                                          v_atpeak=np.nanmedian(reg.ucube.pcubes['1'].parcube[0]))

        res_main_hf_snr = np.nanmax(res_main_hf, axis=0) / best_rms

        # mask out residual with SNR values over the cut threshold
        mask_res = res_main_hf_snr > res_snr_cut
        mask_res = dilation(mask_res)

        cube_res_masked = cube_res.with_mask(~mask_res)
    else:
        cube_res_masked = cube_res

    cube_res_cnv = cnvtool.convolve_sky_byfactor(cube_res_masked, factor=reg.cnv_factor, edgetrim_width=None,
                                                 snrmasked=False, iterrefine=False)

    cube_res_cnv = cube_res_cnv.with_spectral_unit(u.km / u.s, velocity_convention='radio')
    return cube_res_cnv



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

