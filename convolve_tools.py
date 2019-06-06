__author__ = 'mcychen'



import os
import numpy as np
import astropy.io.fits as fits
import FITS_tools
from astropy import units as u
from astropy.stats import mad_std
from skimage.morphology import remove_small_objects, disk, opening, erosion, dilation, remove_small_holes
from spectral_cube import SpectralCube
from radio_beam import Beam
#from reproject import reproject_interp
from astropy.wcs import WCS
#import copy
from astropy.stats import mad_std

#=======================================================================================================================
# utility tools for convolve cubes

def convolve_sky_byfactor(cube, factor, savename, edgetrim_width=5, **kwargs):

    factor = factor * 1.0

    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube)

    if edgetrim_width is not None:
        cube = edge_trim(cube, trim_width=edgetrim_width)

    hdr = cube.header

    # sanity check
    if hdr['CUNIT1'] != hdr['CUNIT2']:
        print "[ERROR]: the axis units for the do not match each other!"
        return None

    beamunit = getattr(u, hdr['CUNIT1'])
    bmaj = hdr['BMAJ'] * beamunit * factor
    bmin = hdr['BMIN'] * beamunit * factor
    pa = hdr['BPA']

    beam = Beam(major=bmaj, minor=bmin, pa=pa)

    # convolve
    cnv_cube = convolve_sky(cube, beam, **kwargs)
    if cnv_cube.fill_value is not np.nan:
        cnv_cube = cnv_cube.with_fill_value(np.nan)

    # regrid the convolved cube
    nhdr = FITS_tools.downsample.downsample_header(hdr, factor=factor, axis=1)
    nhdr = FITS_tools.downsample.downsample_header(nhdr, factor=factor, axis=2)
    nhdr['NAXIS1'] = int(np.rint(hdr['NAXIS1'] / factor))
    nhdr['NAXIS2'] = int(np.rint(hdr['NAXIS2'] / factor))

    # ncube_data = FITS_tools.cube_regrid.regrid_cube(cnv_cube._data, hdr, nhdr, preserve_bad_pixels=True)
    newcube = cnv_cube.reproject(nhdr, order='bilinear')

    if savename is not None:
        # newcube = SpectralCube(ncube_data, header=nhdr)
        newcube.write(savename, overwrite=True)

    return newcube


def convolve_sky(cube, beam, snrmasked=True, iterrefine=True, snr_min=3.0):
    # return the convolved cube in the same gridding as the input
    # note: iterrefine masks data as well

    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube)

    if cube.fill_value is not np.nan:
        cube = cube.with_fill_value(np.nan)

    mask = np.isfinite(cube._data)
    mask = np.any(mask, axis=0)

    if snrmasked:
        planemask = snr_mask(cube, snr_min)
        mask = mask * planemask

    maskcube = cube.with_mask(mask.astype(bool))

    cnv_cube = maskcube.convolve_to(beam)

    if iterrefine:
        # use the convolved cube for new masking
        planemask = snr_mask(cnv_cube, snr_min)
        mask = np.isfinite(cube._data) * planemask
        maskcube = cube.with_mask(mask.astype(bool))
        cnv_cube = maskcube.convolve_to(beam)

    return cnv_cube


def snr_mask(cube, snr_min=3.0, errmappath=None):
    # create a mask around the cube with a snr cut

    if errmappath is not None:
        errmap = fits.getdata(errmappath)

    else:
        # make a quick RMS estimate using median absolute deviation (MAD)
        errmap = mad_std(cube._data, axis=0)
        print "median rms: {0}".format(np.nanmedian(errmap))

    snr = cube.filled_data[:].value / errmap
    peaksnr = np.max(snr, axis=0)

    def default_masking(snr, snr_min=5.0):
        planemask = (snr > snr_min)
        if planemask.size > 100:
            # attempt to remove noisy features
            planemask = erosion(planemask, disk(1))
            planemask = remove_small_objects(planemask, min_size=12)
            # note, dialation is larger than erosion so the foot print is a bit more extended
            planemask = dilation(planemask, disk(2))

        return (planemask)

    planemask = default_masking(peaksnr, snr_min=snr_min)

    return planemask


def edge_trim(cube, trim_width=3):
        # trim the edges by N pixels to guess the location of the peak emission

        mask = np.any(np.isfinite(cube._data), axis=0)
        if mask.size > 100:
            mask = erosion(mask, disk(trim_width))
        mask = np.isfinite(cube._data) * mask

        return cube.with_mask(mask.astype(bool))