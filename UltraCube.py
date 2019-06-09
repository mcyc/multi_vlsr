__author__ = 'mcychen'

#======================================================================================================================#
import os
import numpy as np
import astropy.io.fits as fits
import FITS_tools
from astropy import units as u
from astropy.stats import mad_std
from spectral_cube import SpectralCube
from astropy.wcs import WCS
import pyspeckit
import multiprocessing

import aic

import multi_v_fit as mvf
reload(mvf)

import convolve_tools as cnvtool
reload(cnvtool)

#======================================================================================================================#

class UltraCube(object):

    def __init__(self, cubefile=None, cube=None, **kwargs):
        '''
        # a data frame work to handel multiple component fits and their results
        Parameters
        ----------
        filename : str, optional
            The name of 3D FITS file to load
        '''

        # to hold pyspeckit cubes for fitting
        self.pcubes = {}
        self.chisq_maps = {}
        self.NSamp_map = {}
        self.AICc_maps = {}
        self.master_model_mask = None

        if cubefile is not None:
            self.cubefile = cubefile
            self.load_cube(cubefile)
            return
        else:
            if hasattr(cube, 'spectral_axis'):
                # Load from a SpectralCube instance
                self.cube = cube

        if 'snr_min' in kwargs:
            self.snr_min = kwargs['snr_min']

        if 'rmsfile' in kwargs:
            self.rmsfile = kwargs['rmsfile']

        if 'mask_function' in kwargs:
            self.mask_function = kwargs['mask_function']

        if 'cnv_factor' in kwargs:
            self.cnv_factor = kwargs['cnv_factor']
        else:
            self.cnv_factor = 2



    def load_cube(self, fitsfile):
        # loads SpectralCube
        self.cube = SpectralCube.read(fitsfile)


    def convolve_cube(self, savename=None, edgetrim_width=5):
        # convolved the SpectralCube to a resolution X times the factor specified
        self.cube_cnv = convolve_sky_byfactor(self.cube, self.cnv_factor, savename, edgetrim_width=edgetrim_width)


    def get_cnv_cube(self, filename=None):
        # load the convolve cube if the file exist, create one if otherwise
        if filename is None:
            self.convolve_cube(factor=self.cnv_factor)
        elif os.path.exists(filename):
            self.cube_cnv = SpectralCube.read(fitsfile)
        else:
            print "[WARNING]: the specified file does not exist."


    def fit_cube(self, ncomp, mask_function=None, **kwargs):
        # currently limited to NH3 (1,1) 2-slab fit

        if not 'multicore' in kwargs:
            kwargs['multicore'] = multiprocessing.cpu_count()

        if not 'snr_min' in kwargs:
            kwargs['snr_min'] = self.snr_min


        from collections import Iterable
        if not isinstance(ncomp, Iterable):
            ncomp = [ncomp]

        for nc in ncomp:
            self.pcubes[str(nc)] = mvf.cubefit_gen(self.cube, ncomp=nc, mask_function=mask_function, **kwargs)
            # update model mask
            mod_mask = self.pcubes[str(nc)].get_modelcube() > 0
            self.include_model_mask(mod_mask)


    def include_model_mask(self, mask):
        # update the mask that shows were all the models are non-zero

        if self.master_model_mask is None:
            self.master_model_mask = mask
        else:
            self.master_model_mask = np.logical_or(self.master_model_mask, mask)


    def load_model_fit(self, filename, ncomp):
        self.pcubes[str(ncomp)] = load_model_fit(self.cube, filename, ncomp)
        # update model mask
        mod_mask = self.pcubes[str(ncomp)].get_modelcube() > 0
        self.include_model_mask(mod_mask)


    def get_chisq(self, ncomp, mask=None):
        if mask is None:
            mask = self.master_model_mask
        # note: a mechanism is needed to make sure NSamp is consistient across
        self.chisq_maps[str(ncomp)], self.NSamp_map[str(ncomp)] =\
            get_chisq(self, ncomp, reduced=False, usemask=True, mask=mask)


    def get_reduced_chisq(self, ncomp):
        # no mask is used, and thus is not meant for model comparision
        self.chisq_maps[str(ncomp)], NSamp =\
            get_chisq(self, ncomp, reduced=True, usemask=False, mask=None)


    def get_AICc(self, ncomp, **kwargs):

        compID = str(ncomp)
        if not compID in self.chisq_maps:
            self.get_chisq(ncomp, **kwargs)

        p = ncomp*4
        self.AICc_maps[compID] = get_aic(chisq=self.chisq_maps[compID], p=p, N=self.NSamp_map[compID])

    def get_AICc_likelihood(self, ncomp1, ncomp2):
        return get_AICc_likelihood(self, ncomp1, ncomp2)

#======================================================================================================================#

def fit_cube(cube, **kwargs):
    return mvf.cubefit_gen(cube, **kwargs)


def save_model_fit(pcube, savename, ncomp):
    # specifically save ammonia multi-component model with the right fits header
    mvf.save_pcube(pcube, savename, ncomp)


def load_model_fit(cube, filename, ncomp):
    # currently only loads ammonia multi-component model
    pcube = pyspeckit.Cube(cube=cube)

    # reigster fitter
    linename = 'oneone'
    import ammonia_multiv as ammv

    fitter = ammv.nh3_multi_v_model_generator(n_comp = ncomp, linenames=[linename])
    pcube.specfit.Registry.add_fitter('nh3_multi_v', fitter, fitter.npars)

    pcube.load_model_fit(filename, npars=fitter.npars, fittype='nh3_multi_v')

    return pcube


def convolve_sky_byfactor(cube, factor, savename=None, **kwargs):
    return cnvtool.convolve_sky_byfactor(cube, factor, savename, **kwargs)

#======================================================================================================================#
# statistics tools

def get_cnv_guesses(ncomp, cnv_para_name=None, redo=False):
    #if cnv_para_name is None:
    return None


def get_chisq(ucube, compID, reduced=False, usemask=False, mask=None):

    if isinstance(compID, int):
        compID = str(compID)

    cube = ucube.cube

    if compID is '0':
        modcube = np.zeros(cube.shape)
    else:
        modcube = ucube.pcubes[compID].get_modelcube()

    chi, NSamp = mvf.get_chisq(cube, modcube, expand=20, reduced=reduced, usemask=usemask, mask=mask)
    return chi, NSamp


def get_aic(chisq, p, N=None):
    if N is None:
        return aic.AIC(chisq, p)
    else:
        return aic.AICc(chisq, p, N)


def get_AICc_likelihood(ucube, ncomp1, ncomp2):

    def likelihood(aicc1, aicc2):
        return (aicc1 - aicc2) / 2.0

    if not str(ncomp1) in ucube.NSamp_map:
        ucube.get_AICc(ncomp1)

    if not str(ncomp2) in ucube.NSamp_map:
        ucube.get_AICc(ncomp2)

    NSampEqual = ucube.NSamp_map[str(ncomp1)] == ucube.NSamp_map[str(ncomp2)]
    if np.nansum(~NSampEqual) != 0:
        print("[WARNING]: Number of samples do not match. Recalculating AICc values")
        ucube.get_AICc(ncomp1)
        ucube.get_AICc(ncomp2)

    return likelihood(ucube.AICc_maps[str(ncomp1)], ucube.AICc_maps[str(ncomp2)])




def get_rms(cube, model, expand=20, usemask=True, mask=None):
    '''
    return rms over where no model is fitted
    '''

    import scipy.ndimage as nd
    #model = spectrum.specfit.model

    if usemask:
        if mask is None:
            mask = model > 0
    else:
        mask = ~np.isnan(model)

    #residual = spectrum.specfit.residuals
    residual = cube.filled_data[:].value - model

    # Mask over the region where the fit is non-zero plus a buffer of size set by the expand keyword.
    selem = np.ones(expand, dtype=np.bool)
    selem.shape += (1, 1,)
    mask = nd.binary_dilation(mask, selem)
    #mask = mask.astype(np.float)

    # Now get where the emission is zero and estimate the rms
    # This produces a robust estimate of the RMS along every line of sight:
    diff = residual - np.roll(residual, 2, axis=0)

    '''
    if len(diff) - mask.sum() > 10:
        # only use the mask if there are more than 10 model-free chanels
        diff = diff[~mask]
    '''

    rms = 1.4826 * np.nanmedian(np.abs(diff), axis=0) / 2 ** 0.5
    #print "rms: {}; \t sample size: {}".format(rms, len(diff))

    return rms


def get_residual(cube, model):
    '''
    return rms over where no model is fitted
    '''

    residual = cube.filled_data[:].value - model
    return residual



#======================================================================================================================#

def whatever(self):
    # defining paths to imaged data
    self.OneOneFile = '{2}/{0}/{0}_NH3_{3}_{1}.fits'.format(self.region, self.root, self.cubeDir, self.line_root)
    self.RMSFile = '{2}/{0}/{0}_NH3_{3}_{1}_rms.fits'.format(self.region, self.root, self.cubeDir, self.line_root)
    self.SingVParaFile = '{2}/{0}/{0}_parameter_maps_{1}.fits'.format(self.region, self.rootPara, self.cubeDir)

    cname = "/Users/mcychen/Documents/Data/GAS_NH3/DRMC_rebase3/B59/B59_NH3_11_base_all_rebase3.fits"
    pname = "/Users/mcychen/Desktop/B59_para.fits"
    cnpname = "/Users/mcychen/Desktop/B59_para_cnv.fits"
    import iterative_fit as itf
    cb = itf.cubefit(cname, ncomp=2, multicore=11, snr_min=17, paraname=pname)