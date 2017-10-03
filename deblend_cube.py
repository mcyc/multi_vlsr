__author__ = 'mcychen'

#=======================================================================================================================

import numpy as np
import astropy.io.fits as fits
from spectral_cube import SpectralCube
import astropy.units as u

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
    import matplotlib.pyplot as plt
    # show some errer statistics of the multi-parameter fits
    workDir = "/Users/mcychen/Documents/Data/GAS_NH3"
    paraFile = "{0}/DR1_paraMaps/NGC1333/NGC1333_2vcomp_parameter_maps_DR1_rebase3_clean.fits".format(workDir)
    data, hdr = fits.getdata(paraFile, header = True)

    eV0 = data[8,~np.isnan(data[8])].ravel()
    eV1 = data[12,~np.isnan(data[12])].ravel()
    print np.median(eV0)
    print np.median(eV1)

    dV = np.abs(data[0] - data[4])
    dErrV = np.hypot(data[8], data[12])

    mask = ~np.isnan(dV)
    dV = dV[mask].ravel()
    dErrV = dErrV[mask].ravel()

    mask = dV > dErrV
    dVClean = dV[mask]

    #plt.hist(eV1, bins = 50, cumulative=True, normed= True)

    range = (0, 1.2)

    plt.hist(dV, bins = 30, histtype = 'step', range = range)
    plt.hist(dVClean, bins = 30, histtype = 'step', range = range)
    plt.hist(dErrV, bins = 30, histtype = 'step', range = range)
    plt.legend(["All 2-comp pixels","significant 2-comp pixels", "total 2-comp errors"])
    plt.ylabel("Pixel count")
    plt.xlabel("Velocity differences (km s$^{-1}$)")
    plt.savefig("{0}/DR1_paraMaps/NGC1333/figures/vDiff_histo.pdf".format(workDir))
    plt.show()

def go():
    '''
    dfd
    '''
    workDir = "/Users/mcychen/Documents/Data/GAS_NH3"
    paraFile = "{0}/DR1_paraMaps/NGC1333/NGC1333_2vcomp_parameter_maps_DR1_rebase3_clean.fits".format(workDir)
    cubeFile = "{0}/DR1_rebase3/NGC1333/NGC1333_NH3_11_DR1_rebase3_trim.fits".format(workDir)
    #deblendFile = "{0}/DR1_rebase3/NGC1333/deblended/NGC1333_NH3_11_DR1_rebase3_2vcomp_deblended.fits".format(workDir)
    deblendFile = "{0}/DR1_rebase3/NGC1333/deblended/NGC1333_NH3_11_DR1_rebase3_2vcomp_deblended_fixsigv.fits".format(workDir)

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

    deblend_cube(paraFile, cubeFile, deblendFile, vmin=4.0, vmax=11.0, T_bg = 0.0, sigv_fixed = sigv)
    return None

def deblend_cube(paraFile, cubeRefFile, deblendFile, vmin=4.0, vmax=11.0, T_bg = 0.0, sigv_fixed = None):
    '''
    generated a deblended, two component NH3 (1,1) emission cube assuming the emission is optically thin
    T_bg:
        The backgorund Temperature (default 0 K so there are not background emission in the deblended cube)

    '''

    # open the parameter file
    data_para, hdr_para = fits.getdata(paraFile, header = True)
    n_comp = hdr_para['NAXIS3']/8
    n_para = 4

    # open the reference cube file
    cube = SpectralCube.read(cubeRefFile)
    cube = cube.with_spectral_unit(u.km/u.s,velocity_convention='radio')

    deblend = np.zeros(cube.shape)
    hdr = cube.wcs.to_header()
    spaxis = cube.spectral_axis.value

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
            # nansum the emission from multiple velocity components
            if sigv_fixed is not None:
                T_v = T_mb(spaxis[plane], tex, tau, vlsr, sigv_fixed, T_bg)
            else:
                T_v = T_mb(spaxis[plane], tex, tau, vlsr, sigv, T_bg)
            T_v[np.isnan(T_v)] = 0.0
            deblend[plane,:] += T_v

    newcube = SpectralCube(deblend,cube.wcs,header=hdr)
    slab = newcube.spectral_slab(vmin*u.km/u.s,vmax*u.km/u.s)
    slab.write(deblendFile, overwrite=True)


def trim_cube():
    # trim a deblended cube for testing purpose
    inFile = "/Users/mcychen/Documents/Data/GAS_NH3/DR1_rebase3/NGC1333/deblended/NGC1333_NH3_11_DR1_rebase3_2vcomp_deblended_fixsigv.fits"
    data, hdr = fits.getdata(inFile, header = True)

    if False:
        # the north east filament
        outFile = "/Users/mcychen/Documents/Data/GAS_NH3/DR1_rebase3/test_data/NGC1333_2vcomp_deblended_fixsigv_trim.fits"
        data = data[:,195:240,63:92]

    if True:
        # the main cluster
        outFile = "/Users/mcychen/Documents/Data/GAS_NH3/DR1_rebase3/test_data/NGC1333_2vcomp_deblended_fixsigv_trim_MainCluster.fits"
        data = data[:, 108:150, 85:123]

        #85, 108
        #123, 150

    fits.writeto(outFile, data, hdr)

def test(tex=10.0, tau=0.3, vlsr=0.0, sigv=5.0):
    import matplotlib.pyplot as plt

    v = np.arange(100)*1.0 - 50.0
    T = T_mb(v, tex, tau, vlsr, sigv)
    T += T_mb(v, tex, tau/2.0, vlsr-18.0, sigv*2.0)
    plt.plot(v, T, 'o')
    plt.show()


def T_mb(v, tex, tau, vlsr, sigv, Tbg = 2.73):
    tau_v = tau*np.exp(-(v-vlsr)**2/(2*sigv**2))
    return tex*(1-np.exp(-tau_v)) + Tbg*np.exp(-tau_v)