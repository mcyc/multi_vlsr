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

from os import path

import ammonia_hf_multiv as amhf

#=======================================================================================================================



def get_multiV_models(paraname, refcubename, n_comp = 2, savename = None, snrname = None, rms = 0.15, rmspath = None):
    '''
    Creates a fits file containing the model cubes of individual components stacked into a hypercube
    :param paraname:
    :param refcubename:
    :param n_comp:
    :param savename:
    :param snrname:
    :param rms:
    :param rmspath:
    :return:
    '''

    para, hdr = fits.getdata(paraname, header = True)

    pcube = pyspeckit.Cube(refcubename)
    xarr = pcube.xarr

    cubes = [pcube.cube.copy()
             for i in np.arange(n_comp)]
    cubes = np.array(cubes)
    cubes[:] = np.nan

    # remove the error components
    n_para = n_comp*4
    para = para[:n_para]
    assert para.shape[0] == n_para

    yy,xx = np.indices(para.shape[1:])
    nanvals = np.any(~np.isfinite(para),axis=0)
    isvalid = np.any(para, axis=0) & ~nanvals
    valid_pixels = zip(xx[isvalid], yy[isvalid])

    def model_a_pixel(xy):
        x,y = int(xy[0]), int(xy[1])

        models = [amhf.nh3_vtau_singlemodel(xarr, Tex=tex, tau=tau, xoff_v=vel, width=width)
                  for vel, width, tex, tau in zip(para[::4, y,x], para[1::4, y,x], para[2::4, y,x], para[3::4, y,x])]
        cubes[:,:,y,x] = models

    for xy in ProgressBar(list(valid_pixels)):
        print int(xy[0]), int(xy[1])
        model_a_pixel(xy)

    if savename != None:
        f_name, f_extension = path.splitext(savename)
        for i, data in enumerate(cubes):
            fname = "{0}_v{1}_{2}".format(f_name, i, f_extension)
            model = SpectralCube(data, pcube.wcs, header=pcube.header)
            model.write(fname, overwrite=True)

    if snrname != None:
        # calculate the peak temperature
        Tpeak = np.array([np.nanmax(cube, axis=0)
                 for cube in cubes])

        if rmspath is not None:
            rmsdata = fits.getdata(rmspath)
            if rmsdata.shape == Tpeak[0].shape:
                rms = rmsdata
            else:
                print "[WARNING]: The shape of the rms map ({0}) does not match the shape of the emission map {1}." \
                      " An uniform rms value of: {2} has been adopted instead".format(rmsdata.shape, Tpeak[0].shape, rms)

        snr = Tpeak/rms
        snrfile = fits.PrimaryHDU(data=snr, header=pcube.header)

        for i in np.arange(n_comp*8)+1:
            key = 'PLANE{0}'.format(i)
            if key in hdr:
                hdr.remove(key)

        snrfile.header.set('CDELT3',1)
        snrfile.header.set('CTYPE3','FITPAR')
        snrfile.header.set('PLANE1','SNR_0')
        snrfile.header.set('PLANE2','SNR_1')
        snrfile.header.set('NAXIS3',n_comp*8)
        snrfile.writeto(snrname, overwrite=True)

    return cubes



def get_SNR(paraname, savename = None, rms = 0.15, n_comp = 2):
    '''
    Take a multiple velocity componet fit and produce a signal to noise ratio of the two velocity components
    :param paraname:
    :param savename:
    :param rms:
    :param n_comp:
    :return:
    '''

    para, hdr = fits.getdata(paraname, header = True)
    n_para = n_comp*4

    # remove the error components
    para = para[:n_para]
    assert para.shape[0] == n_para

    yy,xx = np.indices(para.shape[1:])
    nanvals = np.any(~np.isfinite(para),axis=0)
    isvalid = np.any(para, axis=0) & ~nanvals
    valid_pixels = zip(xx[isvalid], yy[isvalid])

    # Create a synthetic X-dimension in km/s
    vres = 0.07
    vpad = 0.5
    vmax, vmin = np.argmax([para[0],para[4]]), np.argmin([para[0],para[4]])
    vmax = vmax + vpad
    vmin = vmin - vpad
    n_samp = (vmax - vmin)/vres

    xarr = np.linspace(vmin, vmax, int(n_samp) + 1, endpoint = True)
    xarr = SpectroscopicAxis(xarr*u.km/u.s, velocity_convention='radio', refX=freq_dict['oneone']*u.Hz).as_unit(u.GHz)

    peakT = np.zeros((n_comp, para.shape[1], para.shape[2]))

    def model_a_pixel(xy):
        x,y = int(xy[0]), int(xy[1])

        models = [amhf.nh3_vtau_singlemodel(xarr, Tex=tex, tau=tau, xoff_v=vel, width=width)
                  for vel, width, tex, tau in zip(para[::4, y,x], para[1::4, y,x], para[2::4, y,x], para[3::4, y,x])]
        peakT[:,y,x] = np.nanmax(models, axis = 1)

    for xy in ProgressBar(list(valid_pixels)):
        print int(xy[0]), int(xy[1])
        model_a_pixel(xy)

    if savename != None:

        for i in np.arange(n_comp*8)+1:
            key = 'PLANE{0}'.format(i)
            if key in hdr:
                hdr.remove(key)

        newfits = fits.PrimaryHDU(data=peakT/rms, header=hdr)
        newfits.header.set('CDELT3',1)
        newfits.header.set('CTYPE3','FITPAR')
        newfits.header.set('PLANE1','SNR_0')
        newfits.header.set('PLANE2','SNR_1')
        newfits.header.set('NAXIS3',n_comp*8)
        newfits.writeto(savename ,overwrite=True)

    return peakT/rms


def get_chisq(cube, model, expand=20, reduced = True, usemask = True, mask = None):
    '''
    cube : SpectralCube

    model: numpy array

    expand : int
        Expands the region where the residual is evaluated by this many channels in the spectral dimension

    reduced : boolean
        Whether or not to return the reduced chi-squared value or not

    usemask: boolean
        Whether or not to mask out some parts of the data.
        If no mask is provided, it masks out samples with model values of zero.

    mask: boolean array
        A mask stating which array elements the chi-squared values are calculated from
    '''

    import scipy.ndimage as nd
    #model = np.zeros(cube.shape)
    cube = cube.with_spectral_unit(u.Hz, rest_value = freq_dict['oneone']*u.Hz)

    if usemask:
        if mask is None:
            mask = model > 0
    else:
        mask = ~np.isnan(model)


    residual = cube.filled_data[:].value-model

    # This calculates chisq over the region where the fit is non-zero
    # plus a buffer of size set by the expand keyword.

    selem = np.ones(expand,dtype=np.bool)
    selem.shape += (1,1,)
    mask = nd.binary_dilation(mask, selem)
    mask = mask.astype(np.float)
    chisq = np.sum((residual * mask)**2, axis=0)

    if reduced:
        chisq /= np.sum(mask, axis=0)

    # This produces a robust estimate of the RMS along every line of sight:
    diff = residual - np.roll(residual, 2, axis=0)
    rms = 1.4826 * np.nanmedian(np.abs(diff), axis=0) / 2**0.5

    chisq /= rms**2

    if reduced:
        # return the reduce chi-squares values
        return chisq

    else:
        # return the ch-squared values and the number of data points used
        return chisq, np.sum(mask, axis=0)


def make_guesses(sigv_para_name, n_comp = 2, tex_guess =10.0, tau_guess = 0.5):
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
    # (may be a little overkill for guesses, but I guess it's okay given that it's not too computationally intensive)
    tau11 = get_singv_tau11(para.copy())

    # construct guesses
    guesses = np.zeros((4*n_comp,)+para.shape[1:])

    if n_comp == 1:
        # provide guesses for where vlsr and sigma has been fitted in DR1 (has a larger footprint than the full 5 para fit)
        guesses[0,:,:] = vlsr           # v centriod
        guesses[1,:,:] = sigma          # linewidth
        guesses[2,:,:] = tex_guess      # T_ex
        guesses[3,:,:] = tau_guess      # tau

        # provide guesses for where tex and tau (and other parameters) has been fitted in DR1
        has_col = para[2] > 0
        guesses[2,has_col] = tex[has_col]               # T_ex
        guesses[3,has_col] = tau11[has_col]             # tau

    if n_comp == 2:
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

    if n_comp > 2:
        print "guesses for > 2 components have not been implemented!"
        return None

    return guesses

def get_singv_tau11(singv_para):
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

    if type(singv_para) == str:
        parcube = fits.getdata(singv_para)

    else:
        parcube = singv_para

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



def cubefit_gen(cube11name, ncomp=2, paraname = None, modname = None, chisqname = None, guesses = None, errmap11name = None,
            multicore = 1, mask_function = None, snr_min=3.0):
    '''
    Perform n velocity component fit on the GAS ammonia 1-1 data.
    (This should be the function to call for all future codes if it has been proven to be reliable)
    Parameters
    ----------
    cube11name : str
        The file name of the ammonia 1-1 cube
    ncomp : int
        The number of components one wish to fit. Default is 2
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
        fitter = amhf.nh3_multi_v_model_generator(n_comp = ncomp)
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
    #idx_peak = np.nanargmax(maskcube.sum(axis=(1,2)))
    #tot_spec = maskcube.sum(axis=(1,2))
    # the mask sometimes could mask out more things than intended... we'll just use the total, unmasked spectrum for now
    tot_spec = cube.sum(axis=(1,2))
    idx_peak = np.nanargmax(tot_spec)
    print "peak T_B: {0}".format(np.nanmax(tot_spec))

    v_atpeak = maskcube.spectral_axis[idx_peak].to(u.km/u.s).value
    print "v_atpeak: {0}".format(v_atpeak)

    # Setup a window around the main hyperfine components
    v_peak_hwidth = 3.0
    vmax = v_atpeak + v_peak_hwidth
    vmin = v_atpeak - v_peak_hwidth

    # Extract the spectrum within the window defined around the main hyperfine components and take moments
    slab = maskcube.spectral_slab(vmin*u.km/u.s, vmax*u.km/u.s)
    m1 = slab.moment1(axis=0).to(u.km/u.s).value
    m2 = (np.abs(slab.moment2(axis=0))**0.5).to(u.km/u.s).value
    # note:  the unit conversion above may fail for spectral_cube version < 0.4.0 (e.g., 0.3.2)
    # Note: due to the hyperfine structures, the NH3 moment 2 overestimates linewidth

    # find the location of the peak signal (to determine the first pixel to fit)
    m0 = slab.moment0(axis=0).value
    m0[np.isnan(m0)] = 0.0 # I'm not sure if this is a good way to get around the sum vs nansum issue
    peakloc = np.nanargmax(m0)
    ymax,xmax = np.unravel_index(peakloc, m0.shape)

    # Make parameter guesses based on the moments [vel, width, tex, tau]
    tex_guess = 10.0
    tau_guess = 0.5

    # Guess linewidth (the current recipe works okay, but potential improvements can be made.
    gs_sig = m2/ncomp
    gs_sig[gs_sig < sigmin] = 0.08 # narrow enough to be purely thermal @ ~10 K

    # there are 4 parameters for each v-component
    gg = np.zeros((ncomp*4,)+pcube.cube.shape[1:])

    if ncomp == 1:
        gg[0,:,:] = m1                 # v0 centriod
        gg[1,:,:] = gs_sig             # v0 width
        gg[2,:,:] = tex_guess          # v0 T_ex
        gg[3,:,:] = tau_guess          # v0 tau

    # using a working recipe (assuming a bright and a faint componet)
    if ncomp == 2:
        gg[0,:,:] = m1 - 0.25*m2       # v0 centriod
        gg[1,:,:] = gs_sig             # v0 width
        gg[2,:,:] = tex_guess          # v0 T_ex
        gg[3,:,:] = tau_guess*0.8      # v0 tau
        gg[4,:,:] = m1 + 0.25*m2       # v1 centriod
        gg[5,:,:] = gs_sig             # v1 width
        gg[6,:,:] = tex_guess          # v1 T_ex
        gg[7,:,:] = tau_guess*0.2      # v1 tau

    # using a generalized receipe that I have not tested clearly could use improvement
    if ncomp > 2:
        for i in range (0, ncomp):
            gg[i,  :,:] = m1+(-1.0+i*1.0/ncomp)*0.5*m2 # v0 centriod (step through a range fo velocities within sigma_v)
            gg[i+1,:,:] = gs_sig             # v0 width
            gg[i+2,:,:] = tex_guess          # v0 T_ex
            gg[i+3,:,:] = tau_guess*0.2      # v0 tau

    if guesses is None:
        guesses = gg

    else:
        # fill in the blanks in the 'guesses'
        # (the two sets of operations below may be a little redundant, but better be safe than sorry I guess)
        has_sigm = guesses[1] > sigmin  + 0.001
        guesses[:,~has_sigm] = gg[:,~has_sigm]
        has_v = guesses[0] != 0.0
        guesses[:,~has_v] = gg[:,~has_v]

    # The guesses should be fine in the first case, but just in case
    guesses[::4][guesses[::4] > vmax] = vmax
    guesses[::4][guesses[::4] < vmin] = vmin

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
                  limitedmax=[True,False,False,False]*ncomp,
                  maxpars=[vmax,0,0,0]*ncomp,
                  limitedmin=[True,True,True,True]*ncomp,
                  minpars=[vmin, sigmin, Tbg, 0]*ncomp,
                  multicore=multicore,
                  **kwargs
                  )
    # Note: use_neighbor_as_guess is currently set to False to ensure the guesses assumes 2 components

    if paraname != None:
        fitcubefile = fits.PrimaryHDU(data=np.concatenate([pcube.parcube,pcube.errcube]), header=pcube.header)
        for i in range (0, ncomp):
            fitcubefile.header.set('PLANE{0}','VELOCITY_{1}'.format(ncomp +1, ncomp))
            fitcubefile.header.set('PLANE{0}','SIGMA_{1}'.format(ncomp +2, ncomp))
            fitcubefile.header.set('PLANE{0}','TEX_{1}'.format(ncomp +3, ncomp))
            fitcubefile.header.set('PLANE{0}','TAU_{1}'.format(ncomp +4, ncomp))
            fitcubefile.header.set('PLANE{0}','eVELOCITY_{1}'.format(ncomp +5, ncomp))
            fitcubefile.header.set('PLANE{0}','eSIGMA_{1}'.format(ncomp +6, ncomp))
            fitcubefile.header.set('PLANE{0}','eTEX_{1}'.format(ncomp +7, ncomp))
            fitcubefile.header.set('PLANE{0}','eTAU_{1}'.format(ncomp +8, ncomp))

        fitcubefile.header.set('CDELT3',1)
        fitcubefile.header.set('CTYPE3','FITPAR')
        fitcubefile.header.set('CRVAL3',0)
        fitcubefile.header.set('CRPIX3',1)
        fitcubefile.writeto(paraname ,overwrite=True)

    if modname != None:
        model = SpectralCube(pcube.get_modelcube(), pcube.wcs, header=cube.header)
        model.write(modname, overwrite=True)

    if chisqname != None:
        chisq = get_chisq(cube, pcube.get_modelcube(), expand=20)
        chisqfile = fits.PrimaryHDU(data=chisq, header=cube.wcs.celestial.to_header())
        chisqfile.writeto(chisqname, overwrite=True)

    return pcube
