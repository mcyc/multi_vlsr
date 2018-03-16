__author__ = 'mcychen'


"""
===================================================
Ammonia inversion transition: Hyperfine-only fitter
===================================================
.. moduleauthor:: Adam Ginsburg <adam.g.ginsburg@gmail.com>
Module API
^^^^^^^^^^
"""
#=======================================================================================================================


import numpy as np
import matplotlib.cbook as mpcb
import copy
import collections

from pyspeckit.mpfit import mpfit
from pyspeckit.spectrum.models import fitter
from pyspeckit.spectrum.models import hyperfine
from pyspeckit.spectrum.models import radex_modelgrid
from pyspeckit.spectrum.models import model
from pyspeckit.spectrum.models.ammonia_constants import (line_names, freq_dict, aval_dict, ortho_dict, voff_lines_dict,
                                                         tau_wts_dict, line_labels)

from astropy import constants
from astropy import units as u
ckms = constants.c.to(u.km/u.s).value


relative_strength_total_degeneracy = collections.defaultdict(lambda: 1)

# only using ammonia (1-1) for the current implementation
line_names = ['oneone']

# sanity check:
for linename in line_names:
    assert len(voff_lines_dict[linename]) == len(tau_wts_dict[linename])

# For each individual inversion line, create a Hyperfine model
# note: the following is built to be generalizable to multiple transitions when line_names contains more than just the
# (1,1) transition name
nh3_vtau = {linename:
            hyperfine.hyperfinemodel({lineid:lineid for lineid,name in
                                      enumerate(voff_lines_dict[linename])},
                                     {lineid:voff for lineid,voff in
                                      enumerate(voff_lines_dict[linename])},
                                     {lineid:freq_dict[linename]*(1-voff/ckms)
                                      for lineid,voff in
                                      enumerate(voff_lines_dict[linename])},
                                     {lineid:tauwt for lineid,tauwt in
                                      enumerate(tau_wts_dict[linename])},
                                     {lineid:1 for lineid,voff in
                                      enumerate(voff_lines_dict[linename])},
                                    )
            for linename in line_names}


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
        models = [nh3_vtau[linename].hyperfine(xarr, Tex=tex, tau=tau, xoff_v=vel, width=width)
                  for vel, width, tex, tau in zip(args[::4], args[1::4], args[2::4], args[3::4])]
        return np.nansum(np.array(models),axis=0)


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
            #fitunits='m/s')
            fitunits='Hz')

    return mod


def nh3_vtau_singlemodel(xarr, Tex, tau, xoff_v, width, linename = 'oneone'):
    # the parameters are in the order of vel, width, tex, tau for each velocity component
    return nh3_vtau[linename].hyperfine(xarr, Tex=Tex, tau=tau, xoff_v=xoff_v, width=width)


#=======================================================================================================================
# "deblended" version of ammonia 11 model

# A roundabout (and lazy) way to create a deblended NH3 (1,1) inversion transition model by making all the indvidual
# transition v_off & frequencies to be the same (i.e., removing the velocity offset)
# note: the following is built to be generalizable to multiple transitions when line_names contains more than just the
# (1,1) transition name
# note 2: the sum of all the tau_values is 2.000003
nh3_vtau_deblended = {linename:
            hyperfine.hyperfinemodel({lineid:lineid for lineid,name in
                                      enumerate(voff_lines_dict[linename])},
                                     {lineid:0.0 for lineid,voff in
                                      enumerate(voff_lines_dict[linename])},
                                     {lineid:freq_dict[linename]*(1.0)
                                      for lineid,voff in
                                      enumerate(voff_lines_dict[linename])},
                                     {lineid:tauwt for lineid,tauwt in
                                      enumerate(tau_wts_dict[linename])},
                                     {lineid:1 for lineid,voff in
                                      enumerate(voff_lines_dict[linename])},
                                    )
            for linename in line_names}


def nh3_vtau_singlemodel_deblended(xarr, Tex, tau, xoff_v, width, linename = 'oneone'):
    # the parameters are in the order of vel, width, tex, tau for each velocity component
    return nh3_vtau_deblended[linename].hyperfine(xarr, Tex=Tex, tau=tau, xoff_v=xoff_v, width=width)