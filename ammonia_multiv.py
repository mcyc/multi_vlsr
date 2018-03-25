__author__ = 'mcychen'


#=======================================================================================================================

import numpy as np

from pyspeckit.spectrum.models import model
from pyspeckit.spectrum.models import ammonia
from pyspeckit.spectrum.models.ammonia_constants import (line_names, freq_dict, aval_dict, ortho_dict,
                                voff_lines_dict, tau_wts_dict)
from pyspeckit.spectrum.models.ammonia_constants import (ckms, h, kb)

#=======================================================================================================================

TCMB = 2.7315 # K

def nh3_multi_v_model_generator(n_comp):
    """
    My attempt to implement 2 componet fits
    Parameters
    ----------
    n_comp : int
        The number of velocity componets to fit
    Returns
    -------
    model : `model.SpectralModel`
        A SpectralModel class build from N different metastable inversion
        hyperfine models
    """
    n_para = n_comp*4
    idx_comp = np.arange(n_comp)

    def nh3_vtau_multimodel(xarr, *args):
        # the parameters are in the order of vel, width, tex, tau for each velocity component
        assert len(args) == n_para
        return ammonia_multi_v(xarr, *args)

    mod = model.SpectralModel(nh3_vtau_multimodel, n_para,
            parnames=[x
                      for ln in idx_comp
                      for x in ('vlsr{0}'.format(ln),
                                'sigma{0}'.format(ln),
                                'tex{0}'.format(ln),
                                'tau{0}'.format(ln))],
            parlimited=[(False,False), (True,False), (True,False), (True,False)]*n_para,
            parlimits=[(0,0), ]*n_para,
            shortvarnames=[x
                           for ln in idx_comp
                           for x in ('v_{{VLSR,{0}}}'.format(ln),
                                     '\\sigma_{{{0}}}'.format(ln),
                                     'T_{{ex,{0}}}'.format(ln),
                                     '\\tau_{{{0}}}'.format(ln))],
            fitunits='Hz')

    return mod


def ammonia_multi_v(xarr, *args):
    # the parameters are in the order of vel, width, tex, tau for each velocity component

    line_names = None

    if xarr.unit.to_string() != 'GHz':
        xarr = xarr.as_unit('GHz')

    if line_names is None:
        line_names = ["oneone"]
    else:
        print "[ERROR]: modeling beyond using just the 1-1 line has yet to be implemented."
        return None

    background_ta = T_antenna(TCMB, xarr.value)
    tau_dict = {}

    # iteratively move through each slabs towards the observer
    for vel, width, tex, tau in zip(args[::4], args[1::4], args[2::4], args[3::4]):

        for linename in line_names:
            tau_dict[linename] = tau

        '''
        model_spectrum = ammonia._ammonia_spectrum(xarr, tex=tex, tau_dict=tau_dict, width=width, xoff_v=vel,
                                                   fortho=fortho, line_names = line_names, background_tb=T_back)
        '''

        model_spectrum = _ammonia_spectrum(xarr, tex=tex, tau_dict=tau_dict, width=width, xoff_v=vel,
                                           line_names=line_names, background_ta=background_ta)

        # Set the spectrum of the current component as the background of the subsequent component
        background_ta = model_spectrum

    return model_spectrum - T_antenna(TCMB, xarr.value)


def _ammonia_spectrum(xarr, tex, tau_dict, width, xoff_v, line_names, background_ta=0.0, fillingfraction=None,
                      return_components=False):
    """
    Helper function: given a dictionary of ammonia optical depths, an excitation tmeperature... etc, and produce a
    spectrum based on a one-slab (i.e., single velocity component model)
    Note: this is a modified version of the _ammonia_spectrum found in pyspeckit/spectrum/models/ammonia.py
    Note2: the final spectrum returned do not have the background emission subtracted

    Parameters
    ----------
    background_ta : float or ndarray
        "Antenna temperature" of the background emission
    Returns
    -------
    model : `model.SpectralModel`
        A SpectralModel class build from N different metastable inversion
        hyperfine models

    """

    # fillingfraction is an arbitrary scaling for the data; the model will be (normal model) * fillingfraction
    if fillingfraction is None:
        fillingfraction = 1.0

    # "runspec" means "running spectrum": it is accumulated over a loop
    runspec = np.zeros(len(xarr))

    if return_components:
        components = []

    for linename in line_names:
        voff_lines = np.array(voff_lines_dict[linename])
        tau_wts = np.array(tau_wts_dict[linename])

        lines = (1-voff_lines/ckms)*freq_dict[linename]/1e9
        tau_wts = tau_wts / (tau_wts).sum()
        nuwidth = np.abs(width/ckms*lines)
        nuoff = xoff_v/ckms*lines

        # tau array
        tauprof = np.zeros(len(xarr))
        for kk,nuo in enumerate(nuoff):
            tauprof_ = (tau_dict[linename] * tau_wts[kk] *
                        np.exp(-(xarr.value+nuo-lines[kk])**2 /
                               (2.0*nuwidth[kk]**2)))
            if return_components:
                components.append(tauprof_)
            tauprof += tauprof_

        T0 = (h*xarr.value*1e9/kb) # "temperature" of wavelength

        if isinstance(tex, dict):

            '''
            runspec = ((T0/(np.exp(T0/tex[linename])-1) -
                        T0/(np.exp(T0/background_tb)-1)) *
                       (1-np.exp(-tauprof)) * fillingfraction + runspec)
            '''
            runspec = ((T0/(np.exp(T0/tex[linename])-1)*(1-np.exp(-tauprof)) +
                        background_ta*np.exp(-tauprof))*fillingfraction
                       + runspec)


        else:
            '''
            runspec = ((T0/(np.exp(T0/tex)-1) -
                        T0/(np.exp(T0/background_tb)-1)) *
                       (1-np.exp(-tauprof)) * fillingfraction + runspec)
            '''
            runspec = ((T0/(np.exp(T0/tex)-1)*(1-np.exp(-tauprof)) +
                        background_ta*np.exp(-tauprof))*fillingfraction
                       + runspec)


    # the return_componets are currently disabled
    if False:
    #if return_components:
        if isinstance(tex, dict):
            term1 = [(T0/(np.exp(T0/tex[linename])-1)-T0/(np.exp(T0/background_tb)-1))
                     for linename in line_names]
        else:
            term1 = (T0/(np.exp(T0/tex)-1)-T0/(np.exp(T0/background_tb)-1))
        return term1*(1-np.exp(-1*np.array(components)))
    else:
        return runspec


def T_antenna(Tbright, nu):
    """
    Calculate antenna temperatures over nu (in GHz)
    """
    T0 = (h*nu*1e9/kb)
    return T0/(np.exp(T0/Tbright)-1)