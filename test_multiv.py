__author__ = 'mcychen'

"""
===================================================
Test the robustness of the multiple components fit
===================================================

^^^^^^^^^^
"""


import numpy as np
import pyspeckit
import astropy.io.fits as fits
from astropy import units as u
from pyspeckit.spectrum.units import SpectroscopicAxis, SpectroscopicAxes
from pyspeckit.spectrum.models.ammonia_constants import freq_dict
from pyspeckit.spectrum.models import ammonia_constants, ammonia

from astropy import wcs
from spectral_cube import SpectralCube

import ammonia_hf_multiv as amhf
reload(amhf)

#=======================================================================================================================

import multi_v_fit as mvf
reload(mvf)

def run():
    #example_spec_fit()

    baseDir = "/Users/mcychen/Documents/Data/GAS_NH3"
    paraDir = "mock_paraMaps"
    cubeDir = "mock_rebase"

    version = "_narrowMidDen"

    cubename = "{0}/{1}/mock_2vcomp{2}_cube.fits".format(baseDir, cubeDir, version)
    realparaname = "{0}/{1}/mock_2vcomp{2}_trueparameter_maps.fits".format(baseDir, paraDir, version)
    peakname = "{0}/{1}/mock_2vcomp{2}_peaktemps.fits".format(baseDir, cubeDir, version)

    if False:
        # generate a fake cube
        #fname = cubename, paraname = realparaname)
        #fake_cube(fname = None, paraname = realparaname)
        fake_cube(peakname=peakname)

    paraname = "{0}/{1}/mock_2vcomp{2}_parameter_maps.fits".format(baseDir, paraDir, version)
    modname = "{0}/{1}/mock_2vcomp{2}_modelcube.fits".format(baseDir, cubeDir, version)

    if False:
        # fit the fake cube with 2 velocity component models
        pcube = example_cube_fit(cubename = cubename, paraname = paraname, modname = modname)
        return pcube

    chiname = "{0}/{1}/mock_2vcomp{2}_chisq.fits".format(baseDir, cubeDir, version)
    if False:
        cube = SpectralCube.read(cubename)
        model = fits.getdata(modname)
        return make_chi_tea(cubename, modname, savename = chiname)

    if False:
        return plot_para_check(realparaname, paraname, chiname, peakname)

    return None


def plot_para_check(name_realp, name_fitp, name_chi, peaktname):
    # check to see how well the fitting routine performed
    para_rl, hdr_rl = fits.getdata(name_realp, header=True)
    para_ft, hdr_ft = fits.getdata(name_fitp, header=True)
    chi, hdr_chi = fits.getdata(name_chi, header=True)
    peakT, hdr_peakT = fits.getdata(peaktname, header=True)
    noise = 0.15 # K km/s; typical of GAS NH3 (1-1) data
    snr = peakT/noise


    swap = para_ft[0] > para_ft[4]
    t = np.copy(para_ft)
    para_ft[:4][:,swap] = para_ft[4:8][:,swap]
    para_ft[4:8][:,swap] = t[:4][:,swap]

    #para_ft[:8][:, swap], para_ft[8:][:, swap] = para_ft[8:][:, swap], para_ft[:8][:, swap]

    #para_ft[0, swap], para_ft[4, swap] = para_ft[4, swap], para_ft[0, swap]
    #para_ft[1, swap], para_ft[5, swap] = para_ft[5, swap], para_ft[1, swap]

    import matplotlib.pyplot as plt
    plt.clf()
    '''
    plt.scatter(para_rl[2], para_ft[0], zorder = 20, s=1)
    plt.scatter(para_rl[2], para_ft[4], zorder = 15, s=1)
    plt.plot(para_rl[2], para_rl[2], c="0.5", zorder = 5)
    '''

    '''
    plt.scatter(para_rl[3], para_ft[1], zorder = 20, s=1)
    plt.scatter(para_rl[3], para_ft[5], zorder = 15, s=1)
    plt.plot(para_rl[3], para_rl[3], c="0.5", zorder = 5)
    '''

    if False:
        #plt.scatter(para_rl[2], para_rl[2] - para_ft[4], zorder = 20, s=1, c=chi, vmin=0.8, vmax=1.3)
        #plt.colorbar()

        plt.scatter(snr[1], para_rl[2] - para_ft[4], zorder = 20, s=1, c=chi, vmin=0.8, vmax=1.3)
        plt.colorbar(label = r'$\chi_{\nu}$')

        #plt.scatter(para_rl[2], para_rl[0] - para_ft[0], zorder = 20, s=1)
        #plt.xlabel(r"Real v$_{lsr}$ (km s$^{-1}$)")
        print snr[0][0,0]
        plt.annotate('SNR is: {0}'.format(int(snr[0][0,0])), xy=(0.5, 0.5))
        plt.title(r"Fitted v$_{lsr}$ of the second component")
        plt.xlabel(r"Peak signal-to-noise ratio")
        plt.ylabel(r"Real v$_{lsr}$ - Fitted v$_{lsr}$ (km s$^{-1}$)")

    if True:
        gmask = snr[1] > 3.0
        plt.hist((para_rl[2] - para_ft[4])[gmask].ravel(), 50, range=(-0.2,0.2), normed=True, histtype = "step")
        plt.hist((para_rl[0] - para_ft[0]).ravel(), 50, range=(-0.2,0.2), normed=True, histtype = "step")


    #plt.scatter(para_rl[3], para_rl[2] - para_ft[4], zorder = 20, s=1)
    plt.show()

    return para_rl, para_ft





def make_chi_tea(cubename, modname, savename = None):
    # calculate chi-squred map based on input models
    cube = SpectralCube.read(cubename)
    model = fits.getdata(modname)
    chisq = chi_sq_map(cube, model)
    if savename != None:
        hdu = fits.PrimaryHDU(chisq, cube.wcs.celestial.to_header())
        hdu.writeto(savename, overwrite=True)

    return chisq

def chi_sq_map(cube, model, expand=20):
    '''
    cube : SpectralCube

    model: numpy array

    expand : int
        Expands the region where the residual is evaluated by this many channels in the spectral dimension
    '''

    import scipy.ndimage as nd
    #model = np.zeros(cube.shape)
    cube = cube.with_spectral_unit(u.Hz, rest_value = freq_dict['oneone']*u.Hz)

    mask = model > 0
    residual = cube.filled_data[:].value-model

    # This calculates chisq over the region where the fit is non-zero
    # plus a buffer of size set by the expand keyword.

    selem = np.ones(expand,dtype=np.bool)
    selem.shape += (1,1,)
    mask = nd.binary_dilation(mask, selem)
    mask = mask.astype(np.float)
    chisq = np.sum((residual * mask)**2, axis=0) / np.sum(mask, axis=0)

    # This produces a robust estimate of the RMS along every line of sight:
    diff = residual - np.roll(residual, 2, axis=0)
    rms = 1.4826 * np.nanmedian(np.abs(diff), axis=0) / 2**0.5

    chisq /= rms**2

    return chisq


def example_cube_fit(cubename = None, paraname = None, modname = None):
    # an example to generate a spectral cube with two velocity components at each pixels

    # generate a fakecube if no cube is provided for the fit
    if cubename == None:
        cube =  fake_cube()
    else:
        cube = SpectralCube.read(cubename)

    # Create a pyspeckit cube
    # the redefinition of xarr is a work-around way of making pyspeckit convention compatible with spectral_cube
    # (see https://github.com/pyspeckit/pyspeckit/issues/86)
    freq11 = freq_dict['oneone']*u.Hz
    xarr = SpectroscopicAxis(cube.spectral_axis, refX=freq11, velocity_convention='radio')
    pcube = pyspeckit.Cube(cube=cube, xarr=xarr)

    # For convenience, convert the X-axis to km/s
    # (WCSLIB automatically converts to m/s even if you give it km/s)
    pcube.xarr.convert_to_unit(u.km/u.s)

    if not 'nh3_2v_11' in pcube.specfit.Registry.multifitters:
        # Use the multi-v model generator to build up a 2 velocity-component model function
        fitter = amhf.nh3_multi_v_model_generator(n_comp = 2)
        # Register the fitter - i.e., tell pyspeckit where it is and how to use it
        pcube.specfit.Registry.add_fitter('nh3_2v_11', fitter, fitter.npars)
        print "number of parameters is {0}".format(fitter.npars)


    # guess the parameter based on the moments [vel, width, tex, tau]
    tex_guess = 10.0
    tau_guess = 0.5

    # set the fit parameter limits
    Tbg = 2.8
    sigmin = 0.04

    # find the position of peak emission in the integrated spectrum over all the pixels
    # this is used to approximate the velocity of the main hyperfine structures in the cube
    idx_peak = cube.sum(axis=(1,2)).argmax()
    v_atpeak = cube.spectral_axis[idx_peak].to(u.km/u.s).value
    # define the half width of the window to moments
    v_peak_hwidth = 3.0
    vmax = v_atpeak + v_peak_hwidth
    vmin = v_atpeak - v_peak_hwidth

    # extract the spectrum within the window defined around the main hyperfine components and take moments
    slab = cube.spectral_slab(vmin*u.km/u.s, vmax*u.km/u.s)

    m1 = slab.moment1(axis=0).to(u.km/u.s).value
    m2 = (np.abs(slab.moment2(axis=0))**0.5).to(u.km/u.s).value # (the second moment is in m^2/s^2, but we want km/s
    # Due to the hyperfines, the NH3 moment overestimates linewidth
    # The v-offset between the two closest main hyperfines is ~ 0.11, and between the brightest main hypefines  ~ 0.32)
    # The following tries to correct for it in the guesses
    #m2 = m2 - 0.32
    #m2[m2<0] = sigmin + 0.01

    '''
    idx = np.arange(100)*10.0/100.0
    vw = []
    lw = []
    la = []
    cw = []
    from scipy import stats
    for i in idx:
        v_atpeak = 0
        v_peak_hwidth = i
        vmax = v_atpeak + v_peak_hwidth
        vmin = v_atpeak - v_peak_hwidth
        slab = cube.spectral_slab(vmin*u.km/u.s, vmax*u.km/u.s)
        vw.append(i)
        method = "wSpecCube"
        lw.append(slab.linewidth_sigma().value[0,0]*1e-3)
        la.append(np.abs(slab.moment2().value[0,0])**0.5*1e-3)
        cw.append(slab.moment1().value[0,0]*1e-3)

        # the following method is rather meaningless...
        #method = "wScipy"
        #lw.append(stats.moment(slab[:,0,0].value, moment=2)*1e-3)
        #cw.append(stats.moment(slab[:,0,0].value, moment=1)*1e-3)


        #cw.append(slab.moment1().value[0,0])
        #lw.append(np.abs(slab.moment2().value[0,0])**0.5*1e-3)
        #lw.append(slab.moment2().value[0,0]*1e-6)

    import matplotlib.pyplot as plt
    vw = np.array(vw)
    lw = np.array(lw)
    cw = np.array(cw)
    plt.plot(vw, lw, zorder = 10)
    plt.plot(vw, cw, zorder = 15)
    plt.plot(vw, la, zorder = 5, c = "0.75")
    plt.axhline(y=0, c='0.5', zorder = 0)
    plt.xlabel(r"Window half-width (km s$^{-1}$)")
    plt.ylabel(r"Moment (km s$^{-1}$)")
    plt.annotate(r'True NH$_3$ $\sigma_{v}$ is (0.1 km s$^{-1}$)', xy=(0.02, 0.7), textcoords = "axes fraction")
    plt.legend(["Moment 2", "Moment 1"])
    plt.savefig("/Users/mcychen/Documents/Data/GAS_NH3/mock_rebase/figures/moments_vs_windowwidthf_{0}.pdf".format(method))
    plt.show()
    '''

    # extract the spectrum within the window defined around the main hyperfine components and take moments
    slab = cube.spectral_slab(vmin*u.km/u.s, vmax*u.km/u.s)

    # First, assemble the guesses:
    # parameters are in the order of [vel, width, tex, tau]

    # guess linewidth
    gs_sig = 0.5*m2
    gs_sig[gs_sig < sigmin] = sigmin + 0.001

    # for 2 v-components, there are 8 parameters in total
    guesses = np.zeros((8,)+pcube.cube.shape[1:])
    guesses[0,:,:] = m1 - 0.25*m2       # v0 centriod
    guesses[1,:,:] = gs_sig             # v0 width
    guesses[2,:,:] = tex_guess          # v0 T_ex
    guesses[3,:,:] = tau_guess          # v0 tau
    guesses[4,:,:] = m1 + 0.25*m2       # v1 centriod
    guesses[5,:,:] = gs_sig             # v1 width
    guesses[6,:,:] = tex_guess          # v1 T_ex
    guesses[7,:,:] = tau_guess/5.0      # v1 tau

    # Do the fit!
    pcube.fiteach(guesses=guesses, # pass in the guess array
            #  tell it where to start the fitting (center pixel in this case)
            #start_from_point=(1,1),
            #use_neighbor_as_guess=True,
            #[v,s,t,t,v,s,t,t]
            limitedmax=[True,False,False,False,True,False,False,False],
            maxpars=[vmax,0,0,0,vmax,0,0,0],
            limitedmin=[True,True,True,True,True,True,True,True],
            minpars=[vmin, sigmin, Tbg, 0, vmin, sigmin, Tbg, 0],
            multicore=3,
            fittype='nh3_2v_11',
            )

    '''
    # re-order the fits, so the lower velocity components always comes first
    r_mask = pcube.parcube[0] > pcube.parcube[4]
    pcube.parcube[:4][r_mask], pcube.parcube[4:][r_mask] = pcube.parcube[4:][r_mask], pcube.parcube[:4][r_mask]
    '''

    fitcubefile = fits.PrimaryHDU(data=np.concatenate([pcube.parcube,pcube.errcube]), header=pcube.header)
    fitcubefile.header.set('PLANE1','VELOCITY_0')
    fitcubefile.header.set('PLANE2','SIGMA_0')
    fitcubefile.header.set('PLANE3','TEX_0')
    fitcubefile.header.set('PLANE4','TAU_0')
    fitcubefile.header.set('PLANE5','VELOCITY_1')
    fitcubefile.header.set('PLANE6','SIGMA_1')
    fitcubefile.header.set('PLANE7','TEX_1')
    fitcubefile.header.set('PLANE8','TAU_1')
    fitcubefile.header.set('PLANE9','eVELOCITY_0')
    fitcubefile.header.set('PLANE10','eSIGMA_0')
    fitcubefile.header.set('PLANE11','eTEX_0')
    fitcubefile.header.set('PLANE12','eTAU_0')
    fitcubefile.header.set('PLANE13','eVELOCITY_1')
    fitcubefile.header.set('PLANE14','eSIGMA_1')
    fitcubefile.header.set('PLANE15','eTEX_1')
    fitcubefile.header.set('PLANE16','eTAU_1')

    fitcubefile.header.set('CDELT3',1)
    fitcubefile.header.set('CTYPE3','FITPAR')
    fitcubefile.header.set('CRVAL3',0)
    fitcubefile.header.set('CRPIX3',1)

    if paraname != None:
        fitcubefile.writeto(paraname ,overwrite=True)

    if modname != None:
        model = SpectralCube(pcube.get_modelcube(), pcube.wcs, header=cube.header)
        model.write(modname, overwrite=True)

    return pcube


def fake_cube(fname = None, paraname = None, peakname = None):
    # Create a fake spectral ammonia (1-1) cube with GAS spectral resolution

    # Create a new WCS object so we can instantiate the SpectralCube
    mywcs = wcs.WCS(naxis=3)

    # GAS spectral resolution in km/s
    spc_res = 0.07

    # Spectral range of the cube
    spc_ulim = 30.0
    spc_llim = -30.0
    n_samp = (spc_ulim - spc_llim)/spc_res

    # Set up a tangent projection (which is normally read from a file.)
    mywcs.wcs.crpix = [1,1,1]
    mywcs.wcs.cdelt = np.array([-0.066667, 0.066667, spc_res*1e3])
    mywcs.wcs.crval = [290.9250, 14.5092, spc_llim*1e3]
    mywcs.wcs.ctype = ["RA---TAN", "DEC--TAN", 'VELO']
    mywcs.wcs.cunit = ['deg', 'deg', 'm/s']

    # Create a synthetic X-dimension in km/s
    xarr = np.linspace(spc_llim, spc_ulim, int(n_samp) + 1, endpoint = True)
    xarr = SpectroscopicAxis(xarr*u.km/u.s, velocity_convention='radio', refX=freq_dict['oneone']*u.Hz).as_unit(u.GHz)

    # First velocity component
    sigm0 = 0.1
    vlsr0 = 0.0

    # upper and lower limits of the second velocity component
    # the 2nd componet sigma and vlsr will be incremented in these ranges
    sigm1_ul = 2.0
    sigm1_ll = sigm0
    vlsr1_ul = vlsr0 + 2.0
    vlsr1_ll = vlsr0

    # the grid dimension (sigma1, vlsr1)
    nx, ny = (30, 30)

    # now create the grid
    sigm1 = np.linspace(sigm1_ll, sigm1_ul, nx)
    vlsr1 = np.linspace(vlsr1_ll, vlsr1_ul, ny)

    data = np.empty((int(n_samp) + 1, ny, nx))
    peakT = np.empty((2, ny, nx))

    # ammonia.ammonia doens't take numpy array, so I'm just going to loop through the values....
    for i, sig in enumerate(sigm1):
        for j, v in enumerate(vlsr1):
            c1 = ammonia.ammonia(xarr, trot=10, ntot=14.5, fortho=0.5, xoff_v=vlsr0, width=sigm0)
            c2 = ammonia.ammonia(xarr, trot=20, ntot=14, fortho=0.5, xoff_v=v, width=sig)
            spectrum = c1 + c2
            data[:,j,i] = spectrum
            peakT[0,j,i], peakT[1,j,i] = c1.max(), c2.max()

    # now add noise
    stddev = 0.15                # K km/s; typical of GAS NH3 (1-1) data
    noise = np.random.normal(loc=0.0, scale=stddev, size=np.shape(data))
    data = data + noise

    cube = SpectralCube(data=data, wcs=mywcs)

    if fname != None:
        cube.write(fname, format='fits', overwrite = True)

    # the write the fake parameters into a cube
    if paraname != None:
        sigm1_v, vlsr1_v = np.meshgrid(sigm1, vlsr1)
        sigm0_v = np.ones(np.shape(sigm1_v))*sigm0
        vlsr0_v = np.ones(np.shape(sigm1_v))*vlsr0
        data = np.array([vlsr0_v, sigm0_v, vlsr1_v, sigm1_v])
        fitcubefile = fits.PrimaryHDU(data=data, header=cube.header)
        fitcubefile.header.set('PLANE1','VELOCITY_0')
        fitcubefile.header.set('PLANE2','SIGMA_0')
        fitcubefile.header.set('PLANE3','VELOCITY_1')
        fitcubefile.header.set('PLANE4','SIGMA_1')
        fitcubefile.header.set('CDELT3',1)
        fitcubefile.header.set('CTYPE3','FITPAR')
        fitcubefile.header.set('CRVAL3',0)
        fitcubefile.header.set('CRPIX3',1)
        fitcubefile.header.set('NAXIS3',4)
        fitcubefile.writeto(paraname ,overwrite=True)

    if peakname != None:
        fitcubefile = fits.PrimaryHDU(data=peakT, header=cube.header)
        fitcubefile.header.set('PLANE1','PEAK_0')
        fitcubefile.header.set('PLANE2','PEAK_0')
        fitcubefile.header.set('CDELT3',1)
        fitcubefile.header.set('CTYPE3','FITPAR')
        fitcubefile.header.set('CRVAL3',0)
        fitcubefile.header.set('CRPIX3',1)
        fitcubefile.header.set('NAXIS3',2)
        fitcubefile.writeto(peakname ,overwrite=True)

    return cube