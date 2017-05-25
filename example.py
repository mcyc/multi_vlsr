__author__ = 'mcychen'

"""
===================================================
Examples of how to do multiple components fit
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

#=======================================================================================================================

import multi_v_fit as mvf
reload(mvf)

def run():
    #example_spec_fit()

    baseDir = "/Users/mcychen/Documents/Data/GAS_NH3"
    paraDir = "mock_paraMaps"
    cubeDir = "mock_rebase"

    # generate a fake cube
    cubename = "{0}/{1}/mock_2vcomp_cube.fits".format(baseDir, cubeDir)
    fake_cube(fname = cubename)

    # fit the fake cube with 2 velocity component models
    paraname = "{0}/{1}/mock_2vcomp_parameter_maps.fits".format(baseDir, paraDir)
    modname = "{0}/{1}/mock_2vcomp_modelcube.fits".format(baseDir, cubeDir)
    pcube = example_cube_fit(cubename = cubename, paraname = paraname, modname = modname)

    return pcube


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

    # find the position of peak emission in the integrated spectrum over all the pixels
    # this is used to approximate the velocity of the main hyperfine structures in the cube
    idx_peak = cube.sum(axis=(1,2)).argmax()
    v_atpeak = cube.spectral_axis[idx_peak].to(u.km/u.s).value
    # define the half width of the window to moments
    v_peak_hwidth = 4.0

    # extract the spectrum within the window defined around the main hyperfine components and take moments
    slab = cube.spectral_slab((v_atpeak - v_peak_hwidth)*u.km/u.s, (v_atpeak + v_peak_hwidth)*u.km/u.s)
    m1 = slab.moment1(axis=0).to(u.km/u.s).value
    m2 = (slab.moment2(axis=0)**0.5).to(u.km/u.s).value # (the second moment is in m^2/s^2, but we want km/s
    #m2 = m2**0.5                                        # due to the hyperfines, the NH3 moment overestimates linewidth


    # First, assemble the guesses:
    # parameters are in the order of [vel, width, tex, tau]

    # for 2 v-components, there are 8 parameters in total
    guesses = np.zeros((8,)+pcube.cube.shape[1:])
    guesses[0,:,:] = m1 - 0.4*m2       # v0 centriod
    guesses[1,:,:] = 0.5*m2             # v0 width
    guesses[2,:,:] = tex_guess          # v0 T_ex
    guesses[3,:,:] = tau_guess          # v0 tau
    guesses[4,:,:] = m1 + 0.4*m2       # v1 centriod
    guesses[5,:,:] = 0.5*m2             # v1 centriod
    guesses[6,:,:] = tex_guess          # v1 T_ex
    guesses[7,:,:] = tau_guess          # v1 tau

    # Do the fit!
    pcube.fiteach(guesses=guesses, # pass in the guess array
            #  tell it where to start the fitting (center pixel in this case)
            #start_from_point=(1,1),
            # Paralellize the fits?
            multicore=4,
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


def fake_cube(fname = None):
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
    sigm1_ul = 0.5
    sigm1_ll = sigm0
    vlsr1_ul = 1.0 + vlsr0
    vlsr1_ll = vlsr0

    # the grid dimension (sigma1, vlsr1)
    nx, ny = (3, 3)

    # now create the grid
    sigm1 = np.linspace(sigm1_ll, sigm1_ul, nx)
    vlsr1 = np.linspace(vlsr1_ll, vlsr1_ul, ny)
    #sigm1_v, vlsr1_v = np.meshgrid(sigm1, vlsr1)

    data = np.empty((int(n_samp) + 1, ny, nx))

    # ammonia.ammonia doens't take numpy array, so I'm just going to loop through the values....
    for i, sig in enumerate(sigm1):
        for j, v in enumerate(vlsr1):
            spectrum = ammonia.ammonia(xarr, trot=10, ntot=15, fortho=0.5, xoff_v=vlsr0, width=sigm0) +\
                       ammonia.ammonia(xarr, trot=20, ntot=14, fortho=0.5, xoff_v=v, width=sig)
            data[:,j,i] = spectrum

    # now add noise
    stddev = 0.15                # K km/s; typical of GAS NH3 (1-1) data
    noise = np.random.normal(loc=0.0, scale=stddev, size=np.shape(data))
    data = data + noise

    cube = SpectralCube(data=data, wcs=mywcs)

    if fname != None:
        cube.write(fname, format='fits', overwrite = True)

    return cube



def example_spec_fit(show = True):
    # An example to generate a NH3 (1-1) spectrum with two velocity components
    # and fit a two velocity components model to it

    # Set spectral resolution in km/s, at 23.7 GHz (0.07 for GBT ammonia)
    spc_rez = 0.07
    # Set spectral range in km/s
    spc_ulim = 30
    spc_llim = -30
    n_samp = (spc_ulim - spc_llim)/spc_rez

    xarr = SpectroscopicAxis(np.linspace(spc_llim, spc_ulim, n_samp)*u.km/u.s,
                             velocity_convention='radio',
                             refX=freq_dict['oneone']).as_unit(u.GHz)


    # Compute a synthetic model made of two velocity components with different properties
    # First, create noise
    stddev = 0.15   # K km/s; typical of GAS NH3 (1-1) data
    noise = np.random.normal(loc=0.0, scale=stddev, size=len(xarr.dxarr))

    synthspec = (ammonia.ammonia(xarr, trot=10, ntot=15, fortho=0.5, xoff_v=0.0, width=0.2) +
                 ammonia.ammonia(xarr, trot=20, ntot=14, fortho=0.5, xoff_v=0.2, width=0.5) +
                 noise)

    # Create the Spectrum object
    spectrum = pyspeckit.Spectrum(xarr=xarr, data=synthspec)

    # Calculate moments of the spectrum (returns background level, amplitude, moment 1, and moment 2)
    mmts = spectrum.moments()

    # Guess the parameter based on the moments [vel, width, tex, tau]
    # Given that I haven't removed the hyperfine components before taking the moment, I'm not sure why the following
    # guess worked as well as it did
    tex_guess = 15
    tau_guess = 0.5
    g0 = [mmts[2] - 0.25*mmts[3], 0.5*mmts[3], tex_guess, tau_guess]
    g1 = [mmts[2] + 0.25*mmts[3], 0.5*mmts[3], tex_guess, tau_guess]
    guesses = g0 + g1

    fit=  mvf.fit_spc(spectrum, guesses = guesses, showPlot = True)
    if show:
        import matplotlib.pyplot as plt
        plt.show()

    return fit

def tt():
    # An example to generate a NH3 (1-1) spectrum with two velocity components
    # and fit a two velocity components model to it

    # Set spectral resolution in km/s, at 23.7 GHz (0.07 for GBT ammonia)
    spc_rez = 0.07
    # Set spectral range in km/s
    spc_ulim = 30
    spc_llim = -30
    n_samp = (spc_ulim - spc_llim)/spc_rez

    xarr = SpectroscopicAxis(np.linspace(spc_llim, spc_ulim, n_samp)*u.km/u.s,
                             velocity_convention='radio',
                             refX=freq_dict['oneone']).as_unit(u.GHz)


    synthspec = ammonia.ammonia(xarr, trot=10, ntot=15, fortho=0.5, xoff_v=0.0, width=0.2, return_tau = True)

    return synthspec