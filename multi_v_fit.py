__author__ = 'mcychen'

import numpy as np
import pyspeckit
import astropy.io.fits as fits
from astropy import units as u
from pyspeckit.spectrum.units import SpectroscopicAxis
from pyspeckit.spectrum.models.ammonia_constants import freq_dict
from pyspeckit.spectrum.models import ammonia

from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar

from skimage.morphology import remove_small_objects,closing,disk,opening

import ammonia_hf_multiv as amhf

#=======================================================================================================================

def make_guesses(sigv_para_name, tex_guess =10.0, tau_guess = 0.5):
    '''
    Make 2 velocity component fit guesses based on the GAS DR1 parameter maps
    Parameters
    ----------
    sigv_para_name : str
        The file name of the DR1 parameter map
    tex_guess: float
        The guess for excitation temperature for where full DR1 model wasn't fitted
    tau_guess: float
        The guess for the ammonia (1-1) optical depth for where full DR1 model wasn't fitted
    Returns
    -------
    guesses : ndarray
        parameter guesses for the 2 velocity component fit
    '''

    para = fits.getdata(sigv_para_name)
    vlsr = para[4]
    sigma = para[3]
    tex = para[1]

    # upper and lower limit of the Tex guesses
    Tmin, Tmax = 2.8, 12.0
    tex[tex < Tmin] = Tmin
    tex[tex > Tmax] = Tmax

    # lower limit of the sigma guess
    sigmin = 0.04

    # Guess linewidth (the current recipe works okay, but potential improvements can be made).
    gs_sig = 0.5*sigma
    gs_sig[gs_sig < sigmin] = sigmin + 0.001


    # get the 1-1 optical depth from the cold_ammonia model
    tau11 = get_singv_tau11(para.copy())

    # construct guesses
    guesses = np.zeros((8,)+para.shape[1:])

    # provide guesses for where vlsr and sigma has been fitted in DR1 (has a larger footprint than the full 5 para fit)
    guesses[0,:,:] = vlsr - 0.25*sigma      # v0 centriod
    guesses[1,:,:] = gs_sig                 # v0 width
    guesses[2,:,:] = tex_guess              # v0 T_ex
    guesses[3,:,:] = tau_guess*0.75         # v0 tau
    guesses[4,:,:] = vlsr + 0.25*sigma      # v1 centriod
    guesses[5,:,:] = gs_sig                 # v1 width
    guesses[6,:,:] = tex_guess              # v1 T_ex
    guesses[7,:,:] = tau_guess*0.25         # v1 tau

    # provide guesses for where tex and tau (and other parameters) has been fitted in DR1
    has_col = para[2] > 0
    guesses[2,has_col] = tex[has_col]                    # v0 T_ex
    guesses[3,has_col] = tau11[has_col]*0.75             # v0 tau
    guesses[6,has_col] = tex[has_col]                    # v1 T_ex
    guesses[7,has_col] = tau11[has_col]*0.25             # v1 tau

    return guesses


def get_singv_tau11(sigv_para):
    '''
    Take a GAS DR1 parameter maps and return optical depth of the 1-1 line.
    Parameters
    ----------
    sigv_para : str or ndarray
        The GAS DR1 parameter cube (i.e., maps), either as a file name or as a 3D ndarray cube.
    Returns
    -------
    tau11 : ndarray
        A map of model optical depths for ammonia (1-1)
    '''

    # Note: the efficiency could benifit from multi-core processing

    if type(sigv_para) == str:
        parcube = fits.getdata(sigv_para)

    else:
        parcube = sigv_para

    # Create a dummy spectral-axis in km/s as a place holder to acquire tau
    xarr = np.linspace(0.0, 10.0, 10, endpoint = True)
    xarr = SpectroscopicAxis(xarr*u.km/u.s, velocity_convention='radio', refX=freq_dict['oneone']*u.Hz).as_unit(u.GHz)

    # set ntot elements with zero values to NaN
    parcube[:,parcube[2] == 0.0] = np.nan

    yy,xx = np.indices(parcube.shape[1:])
    nanvals = np.any(~np.isfinite(parcube),axis=0)
    isvalid = np.any(parcube, axis=0) & ~nanvals
    valid_pixels = zip(xx[isvalid], yy[isvalid])

    tau11 = np.zeros(parcube.shape[1:])

    def model_a_pixel(xy):
        x,y = int(xy[0]), int(xy[1])
        kwargs = {'tkin': parcube[0,y,x],
                  'tex': parcube[1,y,x],
                  'ntot': parcube[2,y,x],
                  'width': parcube[3,y,x],
                  'xoff_v': parcube[4,y,x],
                  'fortho': parcube[5,y,x],
                  'return_tau': True,
                  }
        tau = ammonia.cold_ammonia(xarr, **kwargs)
        tau11[y,x] = tau['oneone']

    for xy in ProgressBar(list(valid_pixels)):
        model_a_pixel(xy)

    return tau11


def cubefit(cube11name, paraname = None, modname = None, guesses = None, errmap11name = None, multicore = 1,
            mask_function = None, snr_min=3.0):
    '''
    Perform 2 velocity component fit on the GAS ammonia 1-1 data.
    Parameters
    ----------
    cube11name : str
        The file name of the ammonia 1-1 cube
    paraname: str
        The output file name of the
    Returns
    -------
    pcube : 'pyspeckit.cubes.SpectralCube.Cube'
        Pyspeckit cube object containing both the fit and the original data cube
    '''

    cube = SpectralCube.read(cube11name)

    pcube = pyspeckit.Cube(cube11name)
    pcube.unit="K"

    # the following check on rest-frequency may not be necessarily for GAS, but better be safe than sorry
    if cube._wcs.wcs.restfrq != np.nan:
        # Specify the rest frequency not present
        cube = cube.with_spectral_unit(u.Hz, rest_value = freq_dict['oneone']*u.Hz)
        cube = cube.with_spectral_unit(u.km/u.s,velocity_convention='radio')

    if pcube.wcs.wcs.restfrq != np.nan:
        # Specify the rest frequency not present
        pcube.xarr.refX = freq_dict['oneone']*u.Hz


    # Register the 2 velocity component fitter
    if not 'nh3_2v_11' in pcube.specfit.Registry.multifitters:
        fitter = amhf.nh3_multi_v_model_generator(n_comp = 2)
        pcube.specfit.Registry.add_fitter('nh3_2v_11', fitter, fitter.npars)
        print "number of parameters is {0}".format(fitter.npars)


    errmap11 = fits.getdata(errmap11name)
    snr = cube.filled_data[:].value/errmap11
    peaksnr = np.max(snr,axis=0)

    # the following function is copied directly from GAS
    def default_masking(snr,snr_min=5.0):
        planemask = (snr>snr_min)
        planemask = remove_small_objects(planemask,min_size=40)
        planemask = opening(planemask,disk(1))
        return(planemask)

    if mask_function is None:
        #from GAS import PropertyMaps as pm
        #planemask = pm.default_masking(peaksnr,snr_min = snr_min)
        planemask = default_masking(peaksnr,snr_min = snr_min)
    else:
        planemask = mask_function(peaksnr,snr_min = snr_min)

    mask = (snr>3)*planemask
    maskcube = cube.with_mask(mask.astype(bool))
    maskcube = maskcube.with_spectral_unit(u.km/u.s,velocity_convention='radio')

    # set the fit parameter limits (consistent with GAS DR1)
    Tbg = 2.8
    sigmin = 0.04

    # Find the velocity of peak emission in the integrated spectrum over all the pixels
    # to estimate where the main hyperfine structures are in the cube
    idx_peak = np.nanargmax(maskcube.sum(axis=(1,2)))
    v_atpeak = maskcube.spectral_axis[idx_peak].to(u.km/u.s).value

    # Setup a window around the main hyperfine components
    v_peak_hwidth = 3.0
    vmax = v_atpeak + v_peak_hwidth
    vmin = v_atpeak - v_peak_hwidth

    # Extract the spectrum within the window defined around the main hyperfine components and take moments
    slab = maskcube.spectral_slab(vmin*u.km/u.s, vmax*u.km/u.s)
    m1 = slab.moment1(axis=0).to(u.km/u.s).value
    m2 = (np.abs(slab.moment2(axis=0))**0.5).to(u.km/u.s).value
    # Note: due to the hyperfine structures, the NH3 moment 2 overestimates linewidth

    # find the location of the peak signal (to determine the first pixel to fit)
    m0 = slab.moment0(axis=0).value
    peakloc = np.nanargmax(m0)
    ymax,xmax = np.unravel_index(peakloc, m0.shape)

    # Make parameter guesses based on the moments [vel, width, tex, tau]
    tex_guess = 10.0
    tau_guess = 0.5

    # Guess linewidth (the current recipe works okay, but potential improvements can be made.
    gs_sig = 0.5*m2
    gs_sig[gs_sig < sigmin] = sigmin + 0.001

    # for 2 v-components, there are 8 parameters in total
    gg = np.zeros((8,)+pcube.cube.shape[1:])
    gg[0,:,:] = m1 - 0.25*m2       # v0 centriod
    gg[1,:,:] = gs_sig             # v0 width
    gg[2,:,:] = tex_guess          # v0 T_ex
    gg[3,:,:] = tau_guess*1.0      # v0 tau
    gg[4,:,:] = m1 + 0.25*m2       # v1 centriod
    gg[5,:,:] = gs_sig             # v1 width
    gg[6,:,:] = tex_guess          # v1 T_ex
    gg[7,:,:] = tau_guess*0.2      # v1 tau

    if guesses == None:
        guesses = gg

    else:
        # fill in the blanks in the 'guesses'
        # (the two sets of operations below may be a little redundant, but better be safe than sorry I guess)
        has_sigm = guesses[1] > sigmin  + 0.001
        guesses[:,~has_sigm] = gg[:,~has_sigm]
        has_v = guesses[0] != 0.0
        guesses[:,~has_v] = gg[:,~has_v]

    # The guesses should be fine in the first case, but just in case
    guesses[0][guesses[0] > vmax], guesses[4][guesses[4] > vmax] = vmax, vmax
    guesses[0][guesses[0] < vmin], guesses[4][guesses[4] < vmin] = vmin, vmin

    # set some of the fiteach() inputs to that used in GAS DR1 reduction
    kwargs = {'integral':False, 'verbose_level':3, 'signal_cut':2}

    '''
    if True:
        # for testing purpose, mask out most of the cube
        # these codes can be removed once the code has been proven to be stable
        a, b = ymax, xmax
        n, m = m1.shape
        r = 1
        y,x = np.ogrid[-a:n-a, -b:m-b]
        mask = x*x + y*y <= r*r
        pcube.cube[:, ~mask] = np.nan
    '''

    # Now fit the cube. (Note: the function inputs are consistent with GAS DR1 whenever possible)
    print('start fit')
    pcube.fiteach(fittype='nh3_2v_11', guesses=guesses,
                  start_from_point=(xmax,ymax),
                  use_neighbor_as_guess=False,
                  #[v,s,t,t,v,s,t,t]
                  limitedmax=[True,False,False,False,True,False,False,False],
                  maxpars=[vmax,0,0,0,vmax,0,0,0],
                  limitedmin=[True,True,True,True,True,True,True,True],
                  minpars=[vmin, sigmin, Tbg, 0, vmin, sigmin, Tbg, 0],
                  multicore=multicore,
                  **kwargs
                  )
    # Note: use_neighbor_as_guess is currently set to False to ensure the guesses assumes 2 components

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
