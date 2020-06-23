__author__ = 'mcychen'

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os
from string import ascii_lowercase
import pyspeckit
import aplpy
import astropy.units as u
from spectral_cube import SpectralCube
from pyspeckit.spectrum.units import SpectroscopicAxis
from pyspeckit.spectrum.models import ammonia
from aplpy.normalize import APLpyNormalize
from astropy.wcs import WCS

# import from parent directory
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import ammonia_multiv as ammv
reload(ammv)


def tau_demo():
    from pyspeckit.spectrum.models.ammonia_constants import freq_dict

    figDir = "/Users/mcychen/Documents/Data/GAS_NH3/other_figures"
    savepath = "{0}/{1}".format(figDir, "NH3_spectrum_tau_demo_vsig1kms.pdf")

    tauAry=np.array([1, 2, 5, 10, 25, 50])
    tex = 6.0
    vlsr = 0
    sigma = 1

    # Create a synthetic X-dimension in km/s
    vmax = 21.0
    vmin = -21.0
    spc_res = 0.07
    n_samp = (vmax - vmin)/spc_res

    xarr = np.linspace(vmin, vmax, int(n_samp) + 1, endpoint = True)
    xarr = SpectroscopicAxis(xarr*u.km/u.s, velocity_convention='radio', refX=freq_dict['oneone']*u.Hz).as_unit(u.GHz)

    # plot spectra
    plt.clf()
    fsize = (8.5,11)
    n_subfig = 6
    f, axarr = multiPlotTemp(numplots = n_subfig, ncols=1, figsize = fsize, polar = False, hspace=0.20, wspace=0.25,
                                xlab=r'v$_{\mathrm{LSR}}$ (km s$^{-1}$)', ylab=r"T$_{\mathrm{MB}}$ (K)",labpad = 15)

    linename = "oneone"
    for i, tau in enumerate(tauAry):
        print i
        print "modeling pixel spectrum {0}".format(i)

        kwargs = {'alpha':0.7}

        mod = ammonia._ammonia_spectrum(xarr.as_unit('GHz'), tex=tex, tau_dict={linename:tau}, width=sigma, xoff_v=vlsr,
                                        fortho=0.0, line_names = [linename])

        axarr[i].plot(xarr.as_unit('km/s').value, mod, **kwargs)

        axarr[i].annotate('{0})'.format(ascii_lowercase[i]), xy = (0.025, 0.8), xycoords='axes fraction')
        axarr[i].annotate("{0} {1}".format(r"$\tau = $", tau), xy = (0.975, 0.8), xycoords='axes fraction'
                          , horizontalalignment='right', zorder = 50)

    f.savefig(savepath)





def test():
    parapath = '/Users/mcychen/Documents/Data/GAS_NH3/DRMC_paraMaps/L1448/paraMaps_MChen/L1448_1vcomp_parameter_maps_base_all_rebase3.fits'
    n_comp = 1
    #obspath = '/Users/mcychen/Documents/Data/GAS_NH3/DRMC_rebase3/L1448/L1448_NH3_11_base_all_rebase3.fits'
    obspath = '/Users/mcychen/Documents/Data/GAS_NH3/DRMC_rebase3/L1448/L1448_NH3_11_all_rebase_multi.fits'
    chipath = '/Users/mcychen/Documents/Data/GAS_NH3/DRMC_paraMaps/L1448/paraMaps_MChen/L1448_1vcomp_chisq_base_all_rebase3.fits'

    savepath = '/Users/mcychen/Desktop/test_spec.pdf'

    yxList = [(105,86),
        (116,99),
        (118,110),
        (125,106),
        (123,125),
        (120,140)]

    # plot spectral fits
    plotMultiSpec(parapath, n_comp, obspath, chipath, yxList, savepath, showSpec = False)

    # plot positions of the spectrum
    mom0path = '/Users/mcychen/Documents/Data/GAS_NH3/DR2_NH3_11_moments/mom0_maps/L1448_NH3_11_base_all_rebase3_mom0.fits'
    savepath = '/Users/mcychen/Desktop/test_map.pdf'
    mapPositions(mom0path, yxList, savepath, showMap = False, cBarLab=r"Integrated Intensity (K km s$^{-1}$)")



def mapPositions(data, yxList, ax=None, figsize=(6,4), stretch=None, vmid=None, clab=None, xoff=16, yoff=2, **kwargs):

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

    if stretch is not None:
        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
        else:
            vmin = np.nanmin(data)
        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
        else:
            vmax = np.nanmax(data)

        if vmid is None:
            alpha = 1000
            vmid = ((alpha + 1) * vmin - vmax) / alpha

        normalizer = APLpyNormalize(stretch='log', vmid=vmid, vmin=vmin, vmax=vmax)
        kwargs['norm'] = normalizer

    im = ax.imshow(data, origin='lower', **kwargs)

    #if cBarLab is not None:
    #    gc.colorbar.set_axis_label_text(cBarLab)

    if clab is not None:
        #if stretch is not None:
        #    plt.colorbar(im, ax=ax, label=clab, norm=normalizer)
        #else:
        plt.colorbar(im, ax=ax, label=clab)
        # match the height of the colorbar with that of the frame

    # show the positions
    for idx, (y,x) in enumerate(yxList):


        bbox_props = dict(boxstyle="circle,pad=0.15", fc="w", ec="b", lw=0, alpha=0.75)
        '''
        ax.annotate(ascii_lowercase[idx], xy=(aidx[1], aidx[0]), xytext=(aidx[1] + oy, aidx[0] + ox), xycoords='data',
                    zorder=120, size=9, horizontalalignment='left', bbox=bbox_props,
                    arrowprops=dict(arrowstyle='-'))
        '''

        ax.plot(x, y, marker='x', markersize=10, markeredgewidth=2, c='red', zorder=100) #markeredgewidth
        ax.annotate(ascii_lowercase[idx] + ')', xy=(x,y), xytext=(x-xoff, y+yoff), xycoords='data',
                    horizontalalignment='center', zorder=50, size=10, bbox=bbox_props,
                    arrowprops=dict(arrowstyle='-'))

    if ax is None:
        return fig



def mapPositions_Aplpy(mappath, yxList, savepath, figure=None, showMap = False, offset = 60.0, cBarLab = None, **kwargs):

    # set default for the map kwargs
    #kwargs = {'vmin':0.7, 'vmax':2.5, 'cmap':"YlGnBu"}
    kwargs_map = {'cmap':"viridis"}#, 'interpolation':"nearest"}
    # update the default with user input
    kwargs_map.update(kwargs)

    gc = aplpy.FITSFigure(mappath, figure=figure)#, subplot=[0.1, 0.1, 0.8, 0.8] )
    gc.show_colorscale(**kwargs_map)
    gc.add_colorbar()
    #gc.colorbar.set_pad(0.1)

    if cBarLab is not None:
        gc.colorbar.set_axis_label_text(cBarLab)

    # specific the annotation offset in arcseconds, then convert it to degrees
    #offset = 12.0
    offset = offset/3600.0

    # show the positions
    for i, (y,x) in enumerate(yxList):
        ra, dec = gc.pixel2world(x,y)
        gc.show_markers(ra, dec, edgecolor='red', facecolor='red', marker='x', s=50, alpha=0.75, zorder = 100)
        gc.add_label(ra + offset, dec + offset, '{0})'.format(ascii_lowercase[i]), zorder = 200, color='black')

    if showMap:
        plt.show()
    #gc.save(savepath)
    return gc



def plotMultiSpec(parapath, n_comp, obspath, chipath, yxList, savepath=None, showSpec = False, vZoomLims = None, lhpath=None,
                  figsize = (5, 6)):
    # plot the chi-squared map and its model for the two component fit

    para, hdr_para = fits.getdata(parapath, header=True)
    map_c, hdr_c = fits.getdata(chipath, header=True)
    if lhpath is not None:
        map_lh, hdr_lh = fits.getdata(lhpath, header=True)

    cube_o = SpectralCube.read(obspath)
    cube_o = cube_o.with_spectral_unit(u.km/u.s, velocity_convention='radio')

    if vZoomLims is not None:
        cube_o = cube_o.spectral_slab(vZoomLims[0]*u.km/u.s, vZoomLims[1]*u.km/u.s)

    # SpectroscopicAxis has the advantage of being able to performed unit conversion automatically
    xarr = SpectroscopicAxis(cube_o.spectral_axis.value, unit = cube_o.spectral_axis.unit,
                             refX=cube_o._header['RESTFRQ'], velocity_convention='radio')

    # remove the error components
    n_para = n_comp*4
    para = para[:n_para]
    assert para.shape[0] == n_para

    linename = "oneone"

    def model_a_pixel(y,x):
        # model individual components
        models = [ammonia._ammonia_spectrum(xarr.as_unit('GHz'), tex=tex, tau_dict={linename:tau}, width=width, xoff_v=vel, fortho=0.0,
                                            line_names = [linename])
                  for vel, width, tex, tau in zip(para[::4, y,x], para[1::4, y,x], para[2::4, y,x], para[3::4, y,x])]
        '''

        models = [ammv.ammonia_multi_v(xarr.as_unit('GHz'), vel, width, tex, tau)
                  for vel, width, tex, tau in zip(para[::4, y,x], para[1::4, y,x], para[2::4, y,x], para[3::4, y,x])]
        '''

        return models

    # plot spectra
    fsize = figsize

    #n_subfig = 6
    n_subfig = len(yxList)

    f, axarr = multiPlotTemp(numplots = n_subfig, ncols=1, figsize = fsize, polar = False, hspace=0.20, wspace=0.25,
                                xlab=r'v$_{\mathrm{LSR}}$ (km s$^{-1}$)', ylab=r"T$_{\mathrm{MB}}$ (K)",labpad = 15)

    for i, (y,x) in enumerate(yxList):
        print i
        print "modeling pixel ({0},{1})".format(x, y)

        kwargs = {'alpha':0.7}

        models = model_a_pixel(y,x)
        spc = cube_o._data[:, y, x]

        # plot the spectrum, the individual components of the model, and the collective model
        axarr[i].plot(xarr.value, spc, c = "0.3", **kwargs)

        tot_model = ammv.ammonia_multi_v(xarr, *para[:,y,x].tolist())
        axarr[i].plot(xarr.value, tot_model, c = "0.0", **kwargs)

        for mod in models:
            axarr[i].plot(xarr.value, mod, **kwargs)

        axarr[i].annotate('{0})'.format(ascii_lowercase[i]), xy = (0.02, 0.7), xycoords='axes fraction')
        #axarr[i].set_ylim((-0.3, np.max(spc) * 1.3))
        axarr[i].set_ylim((np.max(spc)*-0.2), np.max(spc)*1.3)


        if lhpath is not None:
            '''
            axarr[i].annotate("{0} {1}".format(r"$\chi_{\nu}^2 = $", round(map_c[y, x], 2)), xy=(0.975, 0.5),
                              xycoords='axes fraction'
                              , horizontalalignment='right', zorder=50)
            '''

            axarr[i].annotate("{0} {1}".format("$\ln{ \ K^2_1} = $", round(map_lh[y,x],0)), xy = (0.975, 0.67), xycoords='axes fraction'
                              , horizontalalignment='right', zorder = 50)


        else:
            axarr[i].annotate("{0} {1}".format(r"$\chi_{\nu}^2 = $", round(map_c[y, x], 2)), xy=(0.975, 0.8),
                              xycoords='axes fraction'
                              , horizontalalignment='right', zorder=50)

    for ax in axarr[:-1]:
        # remove all the x-labels except for the last subplot
        ax.xaxis.set_major_formatter(plt.NullFormatter())

    if not savepath is None:
        f.savefig(savepath, bbox_inches='tight')
    if showSpec:
        f.show()


########################################################################################################################

# this funciton can be updated to give a more general projection input
def multiPlotTemp(numplots, ncols=2, figsize = None, polar = False, hspace=0.20, wspace=0.25, xlab=None, ylab=None,
                  labpad = 15):
    '''
    Tempate for generating multiplots
    Note: this tempalte was stolen from the anlz_GAS module... I need to go back and clean that place up a bit
    '''

    nrows = numplots/ncols

    maxFigLen = 11.0
    if ncols > nrows:
        factor = maxFigLen/ncols
    else:
        factor = maxFigLen/nrows

    if figsize is None:
        figsize = (ncols*factor, nrows*factor)

    if polar:
        f, axarr = plt.subplots(nrows, ncols, subplot_kw=dict(projection='polar'), figsize = figsize)
    else:
        f, axarr = plt.subplots(nrows, ncols, figsize = figsize)

    f.subplots_adjust(hspace=hspace, wspace=wspace)

    # create a common axis label for all
    axBig = f.add_subplot(111)#, axisbg = "None")

    axBig.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    for indx in axBig.axes.spines.keys():
        axBig.axes.spines[indx].set_color('none')

    if xlab != None:
        axBig.set_xlabel(xlab, labelpad = labpad, size = "large")

    if ylab != None:
        axBig.set_ylabel(ylab, labelpad = labpad, size= "large")

    # make axBig transparent
    axBig.patch.set_alpha(0)

    return f, axarr