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
# note, the following could be simplified a little to account for the fact that current implementation only uses one
# set of inversion lines at a time (e.g., 1-1 only)
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
        The number of velocity componets to vit
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
        # nansum allows the actual model to have less than the specified number of components
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