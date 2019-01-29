__author__ = 'mcychen'


#=======================================================================================================================
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

#=======================================================================================================================

def test():
    workDir = "/Users/mcychen/Documents/Data/GAS_NH3/DRMC_rebase3/L1448"
    cubename = "{0}/L1448_NH3_11_all_rebase_multi.fits".format(workDir)
    #GAS beam is 30.79''
    savename = "{0}/L1448_NH3_11_all_rebase_multi_conv62as.fits".format(workDir)
    return convolve_sky_byfactor(cubename, 2, savename)


def example():

    dir = "/Users/mcychen/Documents/Data/GAS_NH3/mock_rebase/lowC1_xlowC2"
    cubename = "{0}/mock_NH3_11_2vcomp_lowC1_xlowC2_cube.fits".format(dir)

    paraDir = "/Users/mcychen/Documents/Data/GAS_NH3/mock_paraMaps/lowC1_xlowC2/"
    paraname = "{0}/mock_NH3_11_2vcomp_lowC1_xlowC2_parameter_maps_refined.fits".format(paraDir)

    kwargs = {'ncomp':2, 'paraname':paraname, 'modname':None, 'chisqname':None, 'guesses':None, 'errmap11name':None,
              'multicore':3, 'mask_function':None, 'snr_min':3.0, 'linename':"oneone"}

    pcube = cubefit(cubename, downsampfactor=2, **kwargs)
    return pcube


#=======================================================================================================================

def cubefit_wTauTexCnvRef(cubename, singCompRef, downsampfactor=2, **kwargs):
    # uses convolved single component fit as reference to guess Tau and Tex

    gref, hdr_gref = fits.getdata(singCompRef, header=True)
    gTex = gref[2]
    gTau = gref[3]

    ncomp = kwargs['ncomp']
    gg = np.zeros((ncomp*4,)+gTex.shape)
    gg[:] = np.nan

    # using a similar recipe as moment_guesses from multi_v_fit.py

    if ncomp == 1:
        print "[ERROR]: this method is designed for fits with ncomp > 1"
        return None

    if ncomp == 2:
        tau2_frac = 0.25                    # the tau weight of the second component relative to the total fraction
        gg[2,:,:] = gTex                # v0 T_ex
        gg[3,:,:] = gTau*(1-tau2_frac)  # v0 tau
        gg[6,:,:] = gTex*0.8            # v1 T_ex
        gg[7,:,:] = gTau*tau2_frac      # v1 tau

    if ncomp > 2:
        # this recipe has never been tested
        for i in range (0, ncomp):
            gg[i+2,:,:] = gTex*0.75             # T_ex
            gg[i+3,:,:] = gTau/ncomp*0.25       # tau

    cubefit(cubename, downsampfactor=downsampfactor, cnv_guesses=gg, **kwargs)



def cubefit(cubename, downsampfactor=2, cnv_guesses=None, **kwargs):

    root = "conv{0}Xbeam".format(int(np.rint(downsampfactor)))

    if not 'conv_paraname' in kwargs:
        cnv_cubename = "{0}_{1}.fits".format(os.path.splitext(cubename)[0], root)

        print "convolve the cube by a factor of: {0}".format(downsampfactor)
        cnv_cube = convolve_sky_byfactor(cubename, downsampfactor, savename=cnv_cubename, edgetrim_width=None)

        # use the mask of the convolved cube as the mask in which fits will be performed
        mask=cnv_cube.get_mask_array()
        def blank_mask(snr, snr_min=5.0):
            planemask = mask
            return(planemask)

        #kwargs['mask_function']=blank_mask

        # adopt the final kwarg to the convolved fitting
        kwargs_cnv = kwargs.copy()
        kwargs_cnv['paraname'] = "{0}_cnv.fits".format(os.path.splitext(kwargs['paraname'])[0], "parameter_maps")
        # turn the moment edge trim off for now. Something seems to be wrong
        kwargs_cnv['momedgetrim'] = False
        kwargs_cnv['modname'] = None

        if cnv_guesses is not None:
            #kwargs_cnv['saveguess'] = True
            kwargs_cnv['guesses'] = cnv_guesses

        # fit the convolved cube to serve as parameter guesses for the full resolution fitting
        cnv_pcube = mvf.cubefit_gen(cnv_cubename, **kwargs_cnv)

        print "_____________________________________"
        print "fitting convolved cube"
        data_cnv, hdr_cnv = fits.getdata(kwargs_cnv['paraname'], header=True)

    else:
        print "_____________________________________"
        print "fitting convolved cube"
        data_cnv, hdr_cnv = fits.getdata(kwargs['conv_paraname'], header=True)
        # to make kwargs compitable with mvf.cubefit_gen()
        del kwargs['conv_paraname']


    npara = 4
    ncomp = int(data_cnv.shape[0]/npara)/2

    # the target header for the regridding
    cube_hdr = fits.getheader(cubename)
    hdr_final = get_celestial_hdr(cube_hdr)

    '''
    if cnv_guesses is not None:
        # use the provided convolved guesses, particularly Tau & Tex
        # note: may want to make this a separate option from the guess for the convolved fit
        # note: this approach may cause some interpolation issues
        cnv_guesses[cnv_guesses==0] = np.nan
        gmask = np.isfinite(cnv_guesses)
        data_cnv[0:npara*ncomp][gmask] = cnv_guesses[gmask]
    '''


    kwargs['guesses'] = guess_from_cnvpara(data_cnv, hdr_cnv, hdr_final, downsampfactor=2)

    pcube = mvf.cubefit_gen(cubename, **kwargs)

    # write the fitted parameters into a fits file
    mvf.save_pcube(pcube, kwargs['paraname'], ncomp=ncomp)

    return pcube

#=======================================================================================================================

def convolve_sky_byfactor(cube, factor, savename, edgetrim_width=5):

    factor = factor*1.0

    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube)

    if edgetrim_width is not None:
        cube = edge_trim(cube, trim_width=edgetrim_width)

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
    cnv_cube = convolve_sky(cube, beam, snrmasked = True)
    if cnv_cube.fill_value is not np.nan:
        cnv_cube = cnv_cube.with_fill_value(np.nan)

    # regrid the convolved cube
    nhdr = FITS_tools.downsample.downsample_header(hdr, factor=factor, axis=1)
    nhdr = FITS_tools.downsample.downsample_header(nhdr, factor=factor, axis=2)
    nhdr['NAXIS1'] = int(np.rint(hdr['NAXIS1']/factor))
    nhdr['NAXIS2'] = int(np.rint(hdr['NAXIS2']/factor))

    #ncube_data = FITS_tools.cube_regrid.regrid_cube(cnv_cube._data, hdr, nhdr, preserve_bad_pixels=True)
    newcube = cnv_cube.reproject(nhdr, order='bilinear')

    if savename is not None:
        #newcube = SpectralCube(ncube_data, header=nhdr)
        newcube.write(savename, overwrite=True)

    return newcube



def convolve_sky(cube, beam, snrmasked = True, iterrefine= True):

    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube)

    if cube.fill_value is not np.nan:
        cube = cube.with_fill_value(np.nan)

    if snrmasked:
        planemask = snr_mask(cube, snr_min=3.0)
        mask = np.isfinite(cube._data)*planemask
        maskcube = cube.with_mask(mask.astype(bool))

    cnv_cube=maskcube.convolve_to(beam)

    if iterrefine:
        # use the convolved cube for new masking
        planemask = snr_mask(cnv_cube, snr_min=3.0)
        mask = np.isfinite(cube._data)*planemask
        maskcube = cube.with_mask(mask.astype(bool))
        cnv_cube=maskcube.convolve_to(beam)

    return cnv_cube


def snr_mask(cube, snr_min=3.0, errmappath=None):

    if errmappath is not None:
        errmap = fits.getdata(errmappath)

    else:
        # make a quick RMS estimate using median absolute deviation (MAD)
        errmap = mad_std(cube._data, axis=0)
        print "median rms: {0}".format(np.nanmedian(errmap))

    snr = cube.filled_data[:].value/errmap
    peaksnr = np.max(snr,axis=0)

    def default_masking(snr, snr_min=5.0):
        planemask = (snr>snr_min)
        if planemask.size > 100:
            planemask = remove_small_objects(planemask,min_size=40)
            planemask = opening(planemask,disk(1))
        return(planemask)

    planemask = default_masking(peaksnr,snr_min=snr_min)

    return planemask


def edge_trim(cube, trim_width = 3):
    # trim the edges by N pixels to guess the location of the peak emission

    mask = np.any(np.isfinite(cube._data), axis=0)
    if mask.size > 100:
        mask = binary_erosion(mask, disk(trim_width))
    mask = np.isfinite(cube._data)*mask

    return cube.with_mask(mask.astype(bool))


def mask_cleaning(mask):
    # designed to clean a noisy map, with a footprint that is likely slightly larger
    mask = remove_small_objects(mask, min_size=9)
    mask = dilation(mask, disk(1))
    mask = remove_small_holes(mask, 9)
    return mask


def master_mask(pcube):
    # create a 2D mask over where any of the paramater map has finite values
    mask = np.any(np.isfinite(pcube), axis=0)
    mask = mask_cleaning(mask)
    return mask


def guess_from_cnvpara(data_cnv, header_cnv, header_target, downsampfactor=2):
    # a wrapper to make guesses based on the parameters fitted to the convolved data
    npara = 4
    ncomp = int(data_cnv.shape[0]/npara)/2

    data_cnv = data_cnv.copy()
    # clean up the maps based on vlsr errors
    data_cnv = simple_para_clean(data_cnv, ncomp, npara=npara)
    hdr_conv = get_celestial_hdr(header_cnv)
    data_cnv[data_cnv == 0] = np.nan
    data_cnv = data_cnv[0:npara*ncomp]

    mmask = master_mask(data_cnv)

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


def simple_para_clean(pmaps, ncomp, npara=4):
    # clean parameter maps based on their error values

    # remove component with vlsrErr that is number of sigma off from the median as specified below
    std_thres = 2

    pmaps[pmaps == 0] = np.nan

    # loop through each component
    for i in range (0, ncomp):
        # get the STD and Medians of the vlsr errors
        std_vErr = mad_std(pmaps[(i+ncomp)*npara][np.isfinite(pmaps[(i+ncomp)*npara])])
        median_vErr = np.median(pmaps[(i+ncomp)*npara][np.isfinite(pmaps[(i+ncomp)*npara])])

        # remove outlier pixels
        mask = pmaps[(i+ncomp)*npara] > median_vErr + std_vErr*std_thres

        pmaps[i*npara:(i+1)*npara, mask] = np.nan
        pmaps[(i+ncomp)*npara:(i+ncomp+1)*npara, mask] = np.nan

    return pmaps


def refine_guess(map, min=None, max=None, mask=None, disksize=1):
    # refine parameter maps by outlier-fitering, masking, and interpolating
    map = map.copy()

    if min is not None:
        map[map<min] = np.nan
    if max is not None:
        map[map>max] = np.nan

    map = median_filter(map, footprint=disk(disksize))

    if mask is None:
        mask = np.isfinite(map)
        mask = mask_cleaning(mask)

    # interpolate over the dmask footprint
    xline = np.arange(map.shape[1])
    yline = np.arange(map.shape[0])
    X,Y = np.meshgrid(xline, yline)
    itpmask = np.isfinite(map)
    C = intp((X[itpmask],Y[itpmask]), map[itpmask])

    # interpolate over the dmask footprint
    zi = C(X*mask,Y*mask)

    return zi


def regrid(image, header1, header2, dmask=None, method='cubic'):
    # similar to hcongrid from FITS_tools, but uses scipy.interpolate.griddata to interpolate over nan values
    grid1 = get_pixel_mapping(header1, header2)

    xline = np.arange(image.shape[1])
    yline = np.arange(image.shape[0])
    X,Y = np.meshgrid(xline, yline)

    mask = np.isfinite(image)

    if dmask is None:
        dmask = np.ones(grid1[0].shape, dtype=bool)

    return griddata((X[mask],Y[mask]), image[mask], (grid1[1]*dmask, grid1[0]*dmask), method=method, fill_value=np.nan)


def get_celestial_hdr(header):
    # make a new header that only contains celestial (i.e., on-sky) information
    new_hdr = WCS(header).celestial.to_header()
    new_hdr['NAXIS1'] = header['NAXIS1']
    new_hdr['NAXIS2'] = header['NAXIS2']
    return new_hdr
