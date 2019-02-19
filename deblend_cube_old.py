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

def test_fakecube():

    paraFile = "/Users/mcychen/Documents/Data/GAS_NH3/mock_paraMaps/fil_id_tests/mock_2vcomp_noVGrad_paraMaps.fits"
    cubeFile = "/Users/mcychen/Documents/Data/GAS_NH3/mock_rebase/fil_id_tests/ref_empty_modelcube.fits"
    deblendFile = "/Users/mcychen/Documents/Data/GAS_NH3/mock_rebase/fil_id_tests/mock_2vcomp_noVGrad_modelcube.fits"

    # the velocity resolution of each channel
    v_rez = 0.0724
    # the number of pixels to sample a FWHM beam at nyquist limit
    n_pix_nyq_samp = 2.5
    # conversion factor between FWHM and sigma
    fwhm_per_sig = 2.355

    # sigv = 0.1
    # fixed the linewidth at the narrowest allowed by the nyquist sampling
    # the following value is ~0.0769 km/s
    sigv = v_rez*n_pix_nyq_samp/fwhm_per_sig

    deblend_cube(paraFile, cubeFile, deblendFile, vmin=-1.0, vmax=1.0, T_bg = 0.0, sigv_fixed = sigv)

def get_stat():
    # show some error statistics of the multi-parameter fits
    import matplotlib.pyplot as plt


    workDir = "/Users/mcychen/Documents/Data/GAS_NH3"
    #paraFile = "{0}/DR1_paraMaps/NGC1333/two_v_comp/NGC1333_2vcomp_parameter_maps_DR1_rebase3_clean.fits".format(workDir)
    paraFile = "{0}/DR1_paraMaps/NGC1333/two_v_comp/NGC1333_2vcomp_SNR_eachV_DR1_rebase3_clean.fits".format(workDir)
    data, hdr = fits.getdata(paraFile, header = True)

    eV0 = data[8,~np.isnan(data[8])].ravel()
    eV1 = data[12,~np.isnan(data[12])].ravel()
    print np.median(eV0)
    print np.median(eV1)

    dV = np.abs(data[0] - data[4])
    dv_map = dV.copy()
    dErrV = np.hypot(data[8], data[12])

    if False:
        ddata = data.copy()
        swap = data[1] > data[5]
        ddata[1,swap], ddata[5,swap] = ddata[5,swap], ddata[1,swap]
        plt.scatter(ddata[1], ddata[5], s=3)
        plt.ylabel("Sigma_lComp (km s$^{-1}$)")
        plt.xlabel("Sigma_sComp (km s$^{-1}$)")
        plt.savefig("{0}/DR1_paraMaps/NGC1333/figures/sigCompare_scatter.pdf".format(workDir))

    if False:
        dV_clean_map = dV.copy()
        mmask = dV > dErrV
        dV_clean_map[~mmask] = np.nan
        plt.imshow(dV_clean_map, origin="lower", interpolation="nearest", clim=(0.0, 0.8), cmap="jet")
        plt.colorbar()

    mask = ~np.isnan(dV)
    dV = dV[mask].ravel()
    dErrV = dErrV[mask].ravel()

    mask = dV > dErrV
    dVClean = dV[mask]

    #plt.hist(eV1, bins = 50, cumulative=True, normed= True)

    range = (0, 1.2)

    if False:
        plt.hist(dV, bins = 30, histtype = 'step', range = range)
        plt.hist(dVClean, bins = 30, histtype = 'step', range = range)
        plt.hist(dErrV, bins = 30, histtype = 'step', range = range)
        plt.legend(["All 2-comp pixels","significant 2-comp pixels", "total 2-comp errors"])
        plt.ylabel("Pixel count")
        plt.xlabel("Velocity differences (km s$^{-1}$)")
        plt.savefig("{0}/DR1_paraMaps/NGC1333/figures/vDiff_histo.pdf".format(workDir))

    if False:
        plt.scatter(dV, dErrV, s=3)
        plt.plot(dV, dV, c="0.5")
        plt.ylabel("Velocity differences errors (km s$^{-1}$)")
        plt.xlabel("Velocity differences (km s$^{-1}$)")
        plt.savefig("{0}/DR1_paraMaps/NGC1333/figures/vDiff_error_scatter.pdf".format(workDir))

    if True:
        # compare the 1-comp vGrad with the velocity difference between two components

        vg_data = fits.getdata("/Users/mcychen/Documents/Data/GAS_NH3/DR1_paraMaps/NGC1333/vGradMaps/NGC1333_grad_velocity.fits")
        vgrad_map = np.hypot(vg_data[1], vg_data[2])

        mmask = np.isnan(vgrad_map)
        mmask = np.logical_and(mmask, np.isnan(dv_map))

        x = dv_map[~mmask]
        y = vgrad_map[~mmask]

        print np.shape(x)
        print np.shape(y)

        from matplotlib.ticker import NullFormatter
        nullfmt = NullFormatter()         # no labels

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        # start with a rectangular Figure
        plt.figure(1, figsize=(8.5, 6))

        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)

        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        # the scatter plot:
        axScatter.scatter(x, y, s=3)
        axScatter.set_ylabel("Sing. Comp. VGrad (km s$^{-1}$ pix$^{-1}$)")
        axScatter.set_xlabel("Velocity differences (km s$^{-1}$)")

        # now determine nice limits by hand:
        '''
        binwidth = 0.25
        xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
        lim = (int(xymax/binwidth) + 1) * binwidth
        '''

        lim_low = 0
        lim_up_x = 1.2
        lim_up_y = 0.35

        range_x = (lim_low, lim_up_x)
        range_y = (lim_low, lim_up_y)

        axScatter.set_xlim(range_x)
        axScatter.set_ylim(range_y)

        n_bins = 30

        axHistx.hist(x, bins = n_bins, range = range_x)
        axHisty.hist(y, bins = n_bins, range = range_y, orientation='horizontal')

        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())

        #plt.scatter(dv_map, vgrad_mag, s=3)
        #plt.plot(vgrad_mag, vgrad_mag, c="0.5")

        plt.savefig("{0}/DR1_paraMaps/NGC1333/figures/oneCompVgrad_vs_twoCompVDiff_scatter.pdf".format(workDir))


    plt.show()

def go(res_boost = 2.0):
    '''
    :param res_boost:
        the factor to increase the spectral resolution by
    :return:
    '''
    workDir = "/Users/mcychen/Documents/Data/GAS_NH3"
    #paraFile = "{0}/DR1_paraMaps/NGC1333/two_v_comp/NGC1333_2vcomp_DR1_rebase3_clean.fits".format(workDir)
    paraFile = "{0}/DR1_paraMaps/NGC1333/two_v_comp/NGC1333_2vcomp_SNR_eachV_DR1_rebase3_clean.fits".format(workDir)
    cubeFile = "{0}/DR1_rebase3/NGC1333/NGC1333_NH3_11_DR1_rebase3_trim.fits".format(workDir)
    #deblendFile = "{0}/DR1_rebase3/NGC1333/deblended/NGC1333_NH3_11_DR1_rebase3_2vcomp_deblended.fits".format(workDir)
    deblendFile = "{0}/DR1_rebase3/NGC1333/deblended/NGC1333_NH3_11_DR1_rebase3_2vcomp_deblended_fixsigv70mps_2xRes.fits".format(workDir)

    # the velocity resolution of each channel
    v_rez = 0.0724
    # the number of pixels to sample a FWHM beam at nyquist limit
    #n_pix_nyq_samp = 2.5
    n_pix_nyq_samp = 3 # conforming to GAS spatial sampling
    # conversion factor between FWHM and sigma
    fwhm_per_sig = 2.355



    # sigv = 0.1
    # fixed the linewidth at the narrowest allowed by the nyquist sampling
    # the following value is ~0.0769 km/s if n_pix_nyq_samp = 2.5
    sigv = v_rez*n_pix_nyq_samp/fwhm_per_sig
    sigv/res_boost

    # note: the manual sigma_v has to be above Nyquest sampling theorem
    sigv = 0.07 # km/s ; the thermal linewidth of ammonia at 10 K, i.e., 0.0698 km/s (res_boost = 2.0 is needed)

    #deblend_cube(paraFile, cubeFile, deblendFile, vmin=4.0, vmax=11.0, T_bg=0.0, sigv_fixed=sigv, f_spcsamp=res_boost)

    # make a deblended cube at the native resolution
    deblendFile = "{0}/DR1_rebase3/NGC1333/deblended/NGC1333_NH3_11_DR1_rebase3_2vcomp_deblended_nativeSig_2xRes.fits".format(workDir)
    deblend_cube(paraFile, cubeFile, deblendFile, vmin=4.0, vmax=11.0, T_bg=0.0, f_spcsamp=res_boost)

    return None


def deblend_cube(paraFile, cubeRefFile, deblendFile, vmin=4.0, vmax=11.0, T_bg = 0.0, sigv_fixed = None,
                 f_spcsamp = None, tau_wgt = 0.1, convolve=True):

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

    # open the reference cube file
    cube = SpectralCube.read(cubeRefFile)
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

    '''
    xarr = SpectroscopicAxis(mcube.spectral_axis.value, unit = mcube.spectral_axis.unit,
                             refX=mcube._header['RESTFRQ'], velocity_convention='radio')
    '''
    # convert back to an unit that the ammonia hf model can handle (i.e. Hz) without having to create a
    # pyspeckit.spectrum.units.SpectroscopicAxis object (which runs rather slow for model building in comparison)
    mcube = mcube.with_spectral_unit(u.Hz, velocity_convention='radio')
    xarr = mcube.spectral_axis


    # remove the error components
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
        '''
        # reduce the tau just see what's up
        para[3::4, isvalid] = para[3::4, isvalid]/10.0
        '''

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
        '''
        mcube.header['BMAJ'] = cube.header['BMAJ']
        mcube.header['BMIN'] = cube.header['BMIN']
        mcube.header['BPA'] = cube.header['BPA']
        '''
        savename = "{0}_cnv.fits".format(os.path.splitext(deblendFile)[0], ".fits")
        convolve_sky_byfactor(mcube, factor=2, savename=savename, edgetrim_width=None, downsample=False)

    gc.collect()
    return mcube


def deblend_cube_v1(paraFile, cubeRefFile, deblendFile, vmin=4.0, vmax=11.0, T_bg = 0.0, sigv_fixed = None, f_spcsamp = None):
    '''
    generated a deblended, two component NH3 (1,1) emission cube assuming the emission is optically thin
    note this version has some trouble handling background emission
    :param T_bg:
        The backgorund Temperature (default 0 K so there are not background emission in the deblended cube)
    :param f_spcsamp:
        The factor to increase the spectral sampling by, with respect to the resolution of the reference cube

    '''

    # open the parameter file
    data_para, hdr_para = fits.getdata(paraFile, header = True)
    n_comp = hdr_para['NAXIS3']/8
    n_para = 4

    # open the reference cube file
    cube = SpectralCube.read(cubeRefFile)
    cube = cube.with_spectral_unit(u.km/u.s,velocity_convention='radio')

    # trim the cube
    cube = cube.spectral_slab(vmin*u.km/u.s,vmax*u.km/u.s)
    spaxis = cube.spectral_axis.value

    if f_spcsamp is None:
        deblend = np.zeros(cube.shape)
        hdr = cube.wcs.to_header()

    else:
        deblend = np.zeros((cube.shape[0]*int(f_spcsamp), cube.shape[1], cube.shape[2]))
        spaxis = np.interp(np.arange(0, len(spaxis), 1.0/f_spcsamp), np.arange(0, len(spaxis)), spaxis)
        wcs_new = cube.wcs.deepcopy()
        # adjust the spectral reference value
        wcs_new.wcs.crpix[2] = wcs_new.wcs.crpix[2]*int(f_spcsamp)
        hdr = wcs_new.to_header()

    # note, the following lines are highly dependent on how the parameter planes are organized
    for j in range (0, n_comp):
    #for j in range (1,2):
        i = j*n_para
        vlsr = data_para[i,:]
        sigv = data_para[i+1,:]
        tex = data_para[i+2,:]
        tau = data_para[i+3,:]
        #deblend[np.isnan(deblend)] = 0.0

        for plane in np.arange(deblend.shape[0]):
            if sigv_fixed is not None:
                T_v = T_mb(spaxis[plane], tex, tau, vlsr, sigv_fixed, T_bg)
                #T_v =amhf.nh3_vtau_singlemodel_deblended(spaxis[plane], tex, tau, vlsr, sigv_fixed, linename = 'oneone')
            else:
                T_v = T_mb(spaxis[plane], tex, tau, vlsr, sigv, T_bg)
                #T_v =amhf.nh3_vtau_singlemodel_deblended(spaxis[plane], tex, tau, vlsr, sigv, linename = 'oneone')
            # nansum the emission from multiple velocity components
            T_v[np.isnan(T_v)] = 0.0
            deblend[plane,:] += T_v

    newcube = SpectralCube(deblend, cube.wcs, header=hdr)
    newcube.write(deblendFile, overwrite=True)


'''
def test(tex=10.0, tau=0.3, vlsr=0.0, sigv=5.0):
    import matplotlib.pyplot as plt

    v = np.arange(100)*1.0 - 50.0
    T = T_mb(v, tex, tau, vlsr, sigv)
    T += T_mb(v, tex, tau/2.0, vlsr-18.0, sigv*2.0)
    plt.plot(v, T, 'o')
    plt.show()
'''

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