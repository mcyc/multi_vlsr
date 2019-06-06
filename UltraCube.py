__author__ = 'mcychen'

#======================================================================================================================#
import os
import numpy as np
import astropy.io.fits as fits
import FITS_tools
from astropy import units as u
from astropy.stats import mad_std
from skimage.morphology import remove_small_objects, disk, opening, binary_erosion, dilation, remove_small_holes
from spectral_cube import SpectralCube
from radio_beam import Beam
#from reproject import reproject_interp
from astropy.wcs import WCS
#import copy
from astropy.stats import mad_std

from scipy.ndimage.filters import median_filter
from scipy.interpolate import CloughTocher2DInterpolator as intp
from scipy.interpolate import griddata
from FITS_tools.hcongrid import get_pixel_mapping

import multi_v_fit as mvf
reload(mvf)



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

        if cubefile is not None:
            self.cubefile = cubefile
            self.load_fits(cubefile)
            return
        else:
            if hasattr(cube, 'spectral_axis'):
                # Load from a SpectralCube instance
                self.cube = cube

        if 'multicore' in kwargs:
            self.ncpus = kwargs['multicore']

        if 'snr_min' in kwargs:
            self.snr_min = kwargs['snr_min']

        if 'rmsfile' in kwargs:
            self.rmsfile = kwargs['rmsfile']

        if 'mask_function' in kwargs:
            self.mask_function = kwargs['mask_function']

        if 'linename' in kwargs:
            self.linename = kwargs['linename']


    def load_fits(self, fitsfile):
        # loads SpectralCube
        self.cube = SpectralCube.read(fitsfile)


    def convolve_cube(self, factor, savename, edgetrim_width=5):
        convolve_sky_byfactor(self.cube, factor, savename, edgetrim_width)

#======================================================================================================================#

def convolve_sky_byfactor(cube, factor, savename, **kwargs):
    import convolve_tools as cnvtool
    reload(cnvtool)
    return cnvtool.convolve_sky_byfactor(cube, factor, savename, **kwargs)

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