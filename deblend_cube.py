__author__ = 'mcychen'

#=======================================================================================================================

# import external library
import numpy as np
import astropy.io.fits as fits
from spectral_cube import SpectralCube
import astropy.units as u
from astropy.utils.console import ProgressBar
import pyspeckit
from pyspeckit.spectrum.units import SpectroscopicAxis
import gc
import os
from radio_beam import Beam
import FITS_tools

# import from this directory
import ammonia_hf_multiv as amhf
reload(amhf)

#=======================================================================================================================

def deblend_cube(paraFile, cubeRefFile, deblendFile, vmin=4.0, vmax=11.0, T_bg = 0.0, sigv_fixed = None,
                 f_spcsamp = None, tau_wgt = 0.1, convolve=True, deconvolve_linewidth=None):

    # tau_wgt is to account for the fact that the fitted tau parameter is the total tau of all the hyperfines combined
    # a more realistic tau value to adopt will thus be that of the main hyperfines, where the lines becomes optically
    # thick the quickest
    # the total relativel optical depth of the main hyperfines are about 0.5. the two most opaque hyeprfines have weigts
    # of about 0.38 combined (0.14999978 + 0.23333315)
    # The "cleanest" NH3 1,1 satellite hyperfine lines (-19.4117 & -19.55 km/s) has a combine relative weight of ~0.1
    # (i.e., 0.03703694 + 0.07407389)
    # tests show that deblended profile that are optically thin looks the most realist (e.g., smooth), thus the default
    # tau_wgt has been set to 0.1

    para, hdr_para = fits.getdata(paraFile, header = True)
    n_comp = hdr_para['NAXIS3']/8

    # remove the error components
    n_para = n_comp*4
    para = para[:n_para]
    assert para.shape[0] == n_para

    # set linewidth to a fix value upon request
    if sigv_fixed is not None:
        isvalid = np.any(np.isfinite(para), axis=0)
        para[1::4, isvalid] = sigv_fixed

    # deconvolve linewidth if requested (to be used with SCMS for filament ID)
    if deconvolve_linewidth is not None:
        para[1::4] = np.sqrt(para[1::4]**2 - deconvolve_linewidth**2)

    # open the reference cube file
    cube = SpectralCube.read(cubeRefFile)

    kwargs = {'vmin':vmin, 'vmax':vmax, 'f_spcsamp':f_spcsamp, 'tau_wgt':tau_wgt}
    mcube = deblend(para, specCubeRef=cube, **kwargs)

    if deblendFile != None:
        mcube.write(deblendFile, overwrite=True)

    if convolve:
        savename = "{0}_cnv.fits".format(os.path.splitext(deblendFile)[0], ".fits")
        convolve_sky_byfactor(mcube, factor=2, savename=savename, edgetrim_width=None, downsample=False)

    gc.collect()
    return mcube



def deblend(para, specCubeRef, vmin=4.0, vmax=11.0, f_spcsamp = None, tau_wgt = 0.1):

    # tau_wgt is to account for the fact that the fitted tau parameter is the total tau of all the hyperfines combined
    # a more realistic tau value to adopt will thus be that of the main hyperfines, where the lines becomes optically
    # thick the quickest
    # the total relativel optical depth of the main hyperfines are about 0.5. the two most opaque hyeprfines have weigts
    # of about 0.38 combined (0.14999978 + 0.23333315)
    # The "cleanest" NH3 1,1 satellite hyperfine lines (-19.4117 & -19.55 km/s) has a combine relative weight of ~0.1
    # (i.e., 0.03703694 + 0.07407389)
    # tests show that deblended profile that are optically thin looks the most realist (e.g., smooth), thus the default
    # tau_wgt has been set to 0.1

    # open the reference cube file
    cube = specCubeRef
    cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio')

    # trim the cube to the specified velocity range
    cube = cube.spectral_slab(vmin*u.km/u.s,vmax*u.km/u.s)

    # generate an empty SpectralCube to house the deblended cube
    if f_spcsamp is None:
        deblend = np.zeros(cube.shape)
        hdr = cube.wcs.to_header()
    else:
        deblend = np.zeros((cube.shape[0]*int(f_spcsamp), cube.shape[1], cube.shape[2]))
        wcs_new = cube.wcs.deepcopy()
        # adjust the spectral reference value
        wcs_new.wcs.crpix[2] = wcs_new.wcs.crpix[2]*int(f_spcsamp)
        hdr = wcs_new.to_header()

    # retain the beam information
    hdr['BMAJ'] = cube.header['BMAJ']
    hdr['BMIN'] = cube.header['BMIN']
    hdr['BPA'] = cube.header['BPA']

    mcube = SpectralCube(deblend, cube.wcs, header=hdr)

    # convert back to an unit that the ammonia hf model can handle (i.e. Hz) without having to create a
    # pyspeckit.spectrum.units.SpectroscopicAxis object (which runs rather slow for model building in comparison)
    mcube = mcube.with_spectral_unit(u.Hz, velocity_convention='radio')
    xarr = mcube.spectral_axis

    yy,xx = np.indices(para.shape[1:])
    # a pixel is valid as long as it has a single finite value
    isvalid = np.any(np.isfinite(para),axis=0)
    valid_pixels = zip(xx[isvalid], yy[isvalid])

    def model_a_pixel(xy):
        x,y = int(xy[0]), int(xy[1])
        # nh3_vtau_singlemodel_deblended takes Hz as the spectral unit
        models = [amhf.nh3_vtau_singlemodel_deblended(xarr, Tex=tex, tau=tau*tau_wgt, xoff_v=vel, width=width)
                  for vel, width, tex, tau in zip(para[::4, y,x], para[1::4, y,x], para[2::4, y,x], para[3::4, y,x])]

        mcube._data[:,y,x] = np.nansum(np.array(models), axis=0)

    for xy in ProgressBar(list(valid_pixels)):
        model_a_pixel(xy)

    # convert back to km/s in units before saving
    mcube = mcube.with_spectral_unit(u.km/u.s, velocity_convention='radio')
    gc.collect()

    return mcube







def deblend_single(paras, refSpecCube, savename, sigv_fixed=None, convolve=False):

    #cube = SpectralCube.read(ref_cube, ref_wcs)
    cube = refSpecCube
    cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio')
    deblend(paras, cube_ref=cube, deblendFile=savename, vmin=4.0, vmax=11.0, sigv_fixed=sigv_fixed, f_spcsamp=None,
             tau_wgt=0.1, convolve=convolve)




def deblend_old(para, cube_ref, deblendFile, vmin=4.0, vmax=11.0, sigv_fixed = None,
                 f_spcsamp = None, tau_wgt = 0.1, convolve=True):

    n_comp=1

    # tau_wgt is to account for the fact that the fitted tau parameter is the total tau of all the hyperfines combined
    # a more realistic tau value to adopt will thus be that of the main hyperfines, where the lines becomes optically
    # thick the quickest
    # the total relativel optical depth of the main hyperfines are about 0.5. the two most opaque hyeprfines have weigts
    # of about 0.38 combined (0.14999978 + 0.23333315)
    # The "cleanest" NH3 1,1 satellite hyperfine lines (-19.4117 & -19.55 km/s) has a combine relative weight of ~0.1
    # (i.e., 0.03703694 + 0.07407389)
    # tests show that deblended profile that are optically thin looks the most realist (e.g., smooth), thus the default
    # tau_wgt has been set to 0.1

    '''
    para, hdr_para = fits.getdata(paraFile, header = True)
    n_comp = hdr_para['NAXIS3']/8

    # open the reference cube file
    cube = SpectralCube.read(cubeRefFile)
    cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio')
    '''

    # trim the cube to the specified velocity range
    cube = cube_ref.spectral_slab(vmin*u.km/u.s,vmax*u.km/u.s)

    # generate an empty SpectralCube to house the deblended cube
    if f_spcsamp is None:
        deblend = np.zeros(cube.shape)
        hdr = cube.wcs.to_header()
    else:
        deblend = np.zeros((cube.shape[0]*int(f_spcsamp), cube.shape[1], cube.shape[2]))
        wcs_new = cube.wcs.deepcopy()
        # adjust the spectral reference value
        wcs_new.wcs.crpix[2] = wcs_new.wcs.crpix[2]*int(f_spcsamp)
        hdr = wcs_new.to_header()

    mcube = SpectralCube(deblend, cube.wcs, header=hdr)
    # convert back to an unit that the ammonia hf model can handle (i.e. Hz) without having to create a
    # pyspeckit.spectrum.units.SpectroscopicAxis object (which runs rather slow for model building in comparison)
    mcube = mcube.with_spectral_unit(u.Hz, velocity_convention='radio')
    xarr = mcube.spectral_axis

    n_para = n_comp*4
    para = para[:n_para]
    assert para.shape[0] == n_para

    yy,xx = np.indices(para.shape[1:])
    # a pixel is valid as long as it has a single finite value
    isvalid = np.any(np.isfinite(para),axis=0)
    valid_pixels = zip(xx[isvalid], yy[isvalid])

    # set linewidth to a fix value upon request
    if sigv_fixed is not None:
        para[1::4, isvalid] = sigv_fixed

    def model_a_pixel(xy):
        x,y = int(xy[0]), int(xy[1])
        # nh3_vtau_singlemodel_deblended takes Hz as the spectral unit
        models = [amhf.nh3_vtau_singlemodel_deblended(xarr, Tex=tex, tau=tau*tau_wgt, xoff_v=vel, width=width)
                  for vel, width, tex, tau in zip(para[::4, y,x], para[1::4, y,x], para[2::4, y,x], para[3::4, y,x])]

        mcube._data[:,y,x] = np.nansum(np.array(models), axis=0)

    for xy in ProgressBar(list(valid_pixels)):
        model_a_pixel(xy)

    # convert back to km/s in units before saving
    mcube = mcube.with_spectral_unit(u.km/u.s, velocity_convention='radio')

    if deblendFile != None:
        mcube.write(deblendFile, overwrite=True)

    if convolve:
        # retain the beam information
        hdr['BMAJ'] = cube.header['BMAJ']
        hdr['BMIN'] = cube.header['BMIN']
        hdr['BPA'] = cube.header['BPA']
        savename = "{0}_cnv.fits".format(os.path.splitext(deblendFile)[0], ".fits")
        convolve_sky_byfactor(mcube, factor=2, savename=savename, edgetrim_width=None, downsample=False)

    gc.collect()
    return mcube



def T_mb(v, tex, tau, vlsr, sigv, Tbg = 2.73):
    tau_v = tau*np.exp(-(v-vlsr)**2/(2*sigv**2))
    return tex*(1-np.exp(-tau_v)) + Tbg*np.exp(-tau_v)


########################################################################################################################

def convolve_sky_byfactor(cube, factor, savename, edgetrim_width=5, downsample=True):

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
    cnv_cube = convolve_sky(cube, beam)
    if cnv_cube.fill_value is not np.nan:
        cnv_cube = cnv_cube.with_fill_value(np.nan)

    if downsample:
        # regrid the convolved cube
        nhdr = FITS_tools.downsample.downsample_header(hdr, factor=factor, axis=1)
        nhdr = FITS_tools.downsample.downsample_header(nhdr, factor=factor, axis=2)
        nhdr['NAXIS1'] = int(np.rint(hdr['NAXIS1']/factor))
        nhdr['NAXIS2'] = int(np.rint(hdr['NAXIS2']/factor))
        newcube = cnv_cube.reproject(nhdr, order='bilinear')
    else:
        newcube = cnv_cube

    if savename is not None:
        newcube.write(savename, overwrite=True)

    return newcube


def convolve_sky(cube, beam, mask = None):

    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube)

    if cube.fill_value is not np.nan:
        cube = cube.with_fill_value(np.nan)

    if mask is not None:
        maskcube = cube.with_mask(mask.astype(bool))
        cnv_cube = maskcube.convolve_to(beam)

    else:
        cnv_cube=cube.convolve_to(beam)

    return cnv_cube


def edge_trim(cube, trim_width = 3):
    from skimage.morphology import binary_erosion, disk
    # trim the edges by N pixels to guess the location of the peak emission

    mask = np.any(np.isfinite(cube._data), axis=0)
    if mask.size > 100:
        mask = binary_erosion(mask, disk(trim_width))
    mask = np.isfinite(cube._data)*mask

    return cube.with_mask(mask.astype(bool))


########################################################################################################################
# wrappers