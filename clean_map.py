__author__ = 'mcychen'

import os
import numpy as np
import astropy.io.fits as fits
from matplotlib import pyplot as plt
import multi_v_fit as mvf
from skimage import morphology

"""
===================================================
For cleaning up multi_v_parameters
===================================================
.. moduleauthor:: Mike Chen <mcychen@uvic.ca>
Module API
^^^^^^^^^^
"""
#=======================================================================================================================

def tt():
    baseDir = "/Users/mcychen/Documents/Data/GAS_NH3/DR1_rebase3/NGC1333"
    #aiccname = "{0}/two_v_comp/NGC1333_2v1comp_aicc.fits".format(baseDir)
    aiccname = "{0}/two_v_comp/NGC1333_2v1comp_chisq.fits".format(baseDir)

    aicc = fits.getdata(aiccname)
    #mask = aicc[0] < aicc[1]
    plt.imshow(aicc[0]-aicc[1], origin="lower", interpolation="nearest")
    #plt.imshow(aicc[0]/aicc[1], origin="lower", interpolation="nearest")
    #plt.imshow(aicc[1, mask], origin="lower", interpolation="nearest")

    ratio = (aicc[0]/aicc[1]).ravel()
    ratio = ratio[~np.isnan(ratio)]

    #plt.hist(ratio)
    plt.show()

def clean_reg(reg = "NGC1333"):
    # clean the 2 component fit of a region using the defaults

    workDir = "/Users/mcychen/Documents/Data/GAS_NH3/DR1_paraMaps/{0}/two_v_comp".format(reg)
    if not os.path.exists(workDir):
        os.makedirs(workDir)

    oriname = "{0}/{1}_2vcomp_parameter_maps_DR1_rebase3.fits".format(workDir, reg)
    newname = "{0}/{1}_2vcomp_SNR_eachV_DR1_rebase3_clean.fits".format(workDir, reg)
    snrname = "{0}/{1}_2vcomp_SNR_eachV_DR1_rebase3.fits".format(workDir, reg)

    clean(oriname, newname, snrname)

def run(reg = "NGC1333"):
    # run some cleaning test runs

    if reg == "NGC1333":
        workDir = "/Users/mcychen/Documents/Data/GAS_NH3/DR1_paraMaps/{0}".format(reg)
        oriname = "{0}/NGC1333_2vcomp_parameter_maps_DR1_rebase3.fits".format(workDir, reg)
        snrname = "{0}/NGC1333_2vcomp_snr_maps_DR1_rebase3.fits".format(workDir, reg)
        newname = "{0}/NGC1333_2vcomp_parameter_maps_DR1_rebase3_clean.fits".format(workDir, reg)
        #newname = "{0}/NGC1333_2vcomp_parameter_maps_DR1_rebase3_clean2.fits".format(workDir, reg)

        chiname = "{0}/NGC1333_2vcomp_chisq_DR1_rebase3.fits".format(workDir, reg)

    else:
        return None

    if False:
        make_SNR_map(oriname, snrname)

    #clean(oriname, newname)
    #test(newname)

    #test(oriname, snrname)
    clean(oriname, newname, snrname)
    #test(newname)



def test(fname, snrname = None):

    para = fits.getdata(fname)

    v1, sig1, tex1, tau1 = para[:4]
    v2, sig2, tex2, tau2 = para[4:8]
    v1_err, sig1_err, tex1_err, tau1_err = para[8:12]
    v2_err, sig2_err, tex2_err, tau2_err = para[12:]

    plt.clf()

    if snrname == None:
        #plt.scatter(v1, v2, s =1)
        #plt.scatter(sig1, sig2, s=1)
        '''
        plt.scatter(v1, sig1, s =1)
        plt.scatter(v2, sig2, s =1)
        '''
        if False:
            # this figure may suggest things being turbulent?
            plt.scatter(sig1, sig2, s =1, c=np.abs(v1-v2))

    else:
        snr = fits.getdata(snrname)

        if False:
            snr_min = 3.0
            mask = np.logical_and(snr[0] > snr_min, snr[1] > snr_min)
            #plt.scatter(v1[mask], v2[mask], s=1, c=snr[0][mask])
            plt.scatter(snr[0][mask], v1[mask] - v2[mask], s=1)
        if False:
            plt.scatter(snr[0], sig1, s=1)
            plt.scatter(snr[1], sig2, s=1)
        if True:
            sm = 5.0
            mask = np.logical_and(snr[0]>sm, snr[1]>sm)
            plt.plot(v1[mask], v1[mask], c='0.5')
            plt.scatter(v1[mask], v2[mask], s =1)


    if False:
        mask = np.abs(v1 - v2) < np.hypot(v1_err, v2_err)
        mask = np.logical_and(mask, np.abs(sig1 - sig2) < np.hypot(sig1_err, sig2_err))
        #mask = np.abs(v1 - v2) < np.min([v1_err, v2_err], axis = 0)
        #mask = np.logical_and(mask, np.abs(sig1 - sig2) < np.min([sig1_err, sig2_err], axis=0))
        plt.scatter(np.abs(sig1- sig2)[mask], np.abs(v1- v2)[mask], s = 1)

    #plt.scatter(sig1, sig1_err, s=1)
    plt.show()

def clean(readname, savename, snrname, fill_plane1 = False, mergesim = False, rm_sml_obj = True, sort_method = "None"):
    '''
    A good recipe so far:
    1. SNR culling
    2. Place larger line width component on the second plane
    3. Merge pixels with similar velocity and linewdiths into one, and place it on the 1st plane
    '''


    para, hdr_para = fits.getdata(readname, header = True)
    snr, hdr_snr = fits.getdata(snrname, header = True)

    v1, sig1, tex1, tau1 = para[:4]
    v2, sig2, tex2, tau2 = para[4:8]
    v1_err, sig1_err, tex1_err, tau1_err = para[8:12]
    v2_err, sig2_err, tex2_err, tau2_err = para[12:]

    # remove the zero values in vlsr
    mask_all = np.logical_or(v1 == 0, v2 == 0)

    # remove pixels with low SNR
    snr_min = 3.0
    mask_1 = snr[0] < snr_min
    mask_2 = snr[1] < snr_min
    # mask out first velocity component
    para[:4, mask_1], para[8:12, mask_1] = np.nan, np.nan
    # mask out second velocity component
    para[4:8, mask_2], para[12:, mask_2] = np.nan, np.nan


    # remove pixels with line width that are outside the expected physical range
    sig_min = 0.05
    mask_1 = sig1 <= sig_min
    mask_2 = sig2 <= sig_min
    sig_max = 5.0           # all the hyperfines starts to blend together at this point
    mask_1 = np.logical_or(mask_1, sig1 >= sig_max)
    mask_2 = np.logical_or(mask_2, sig2 >= sig_max)

    # remove pixels where the error in the line width is larger than linewidth itself
    mask_1 = np.logical_or(mask_1, sig1 <= sig1_err)
    mask_2 = np.logical_or(mask_2, sig2 <= sig2_err)

    # mask out all maps
    para[:, mask_all] = np.nan

    # mask out first velocity component
    para[:4, mask_1], para[8:12, mask_1] = np.nan, np.nan

    # mask out second velocity component
    para[4:8, mask_2], para[12:, mask_2] = np.nan, np.nan

    '''
    # place components with larger (f_sig times) linewidths
    # (i.e., what is more likely the ambient gas)
    f_sig = 50.0
    swap = sig1 > sig2*f_sig
    #swap = sig1 > sig2 + sig1_err + sig1_err #+ np.hypot(sig1_err, sig1_err)

    ttt = para.copy()
    para[:4, swap], para[8:12, swap] = para[4:8, swap], para[12:, swap]
    para[4:8, swap], para[12:, swap] = ttt[:4, swap], ttt[8:12, swap]
    '''

    #if SNRSort:
    if sort_method == "SNR":
        swap = snr[0] < snr[1]

    #elif purewidthSort:
    elif sort_method == "pure_width":
        # place components with larger linewidths into the second componet
        # (i.e., what is more likely the ambient gas)
        f_sig = 1.0
        #swap = np.logical_and(sig1 > sig2*f_sig, np.abs(v1 - v2) < np.nanmin([sig1, sig2], axis = 0))
        #swap = np.logical_and(sig1 > sig2 + sig1_err + sig1_err, np.abs(v1 - v2) < np.nanmin([sig1, sig2], axis = 0))
        swap = sig1 > sig2+np.hypot(sig1_err, sig1_err)

    #else:
    elif sort_method == "sim_v_width":
        # place components with larger linewidths but "similar velocities" into the second componet
        # (i.e., what is more likely the ambient gas)
        f_sig = 1.0
        #swap = np.logical_and(sig1 > sig2*f_sig, np.abs(v1 - v2) < np.nanmin([sig1, sig2], axis = 0))
        #swap = np.logical_and(sig1 > sig2 + sig1_err + sig1_err, np.abs(v1 - v2) < np.nanmin([sig1, sig2], axis = 0))
        swap = np.logical_and(sig1 > sig2+np.hypot(sig1_err, sig1_err)*0.0, np.abs(v1 - v2) < np.nanmin([sig1,sig2], axis=0))


    if sort_method != "None":
        ttt = para.copy()
        para[:4, swap], para[8:12, swap] = para[4:8, swap], para[12:, swap]
        para[4:8, swap], para[12:, swap] = ttt[:4, swap], ttt[8:12, swap]

    if fill_plane1:
        # fill in the blanks in component 1 by moving the component 2 pixels "forward"
        # note, the linewdith map does not look very coherent with this approach
        swap = np.logical_and(np.isnan(v1),  ~np.isnan(v2))
        ttt = para.copy()
        para[:4, swap], para[8:12, swap] = para[4:8, swap], para[12:, swap]
        para[4:8, swap], para[12:, swap] = ttt[:4, swap], ttt[8:12, swap]

    if mergesim:
        # merge pixels with the same v and sig within the estimated uncertainties, adopting their mean values
        # note: a signal to noise criteria may be a better way to determine which values to adopt
        merge = np.abs(v1 - v2) < np.hypot(v1_err, v2_err)
        merge = np.logical_and(merge, np.abs(sig1 - sig2) < np.hypot(sig1_err, sig2_err))
        para[:4, merge], para[8:12, merge] = np.mean([para[4:8, merge], para[:4, merge]], axis=0),\
                                           np.mean([para[12:, merge], para[8:12, merge]], axis=0)
        para[4:8, merge], para[12:, merge] = np.nan, np.nan


    if rm_sml_obj:
        # remove islated pixels of a certain size
        mask = ~np.isnan(para)
        mask = mask.sum(axis=0)
        mask = mask/mask
        mask = mask.astype(bool)
        #plt.imshow(mask, origin="lower")
        #plt.show()
        clean_mask = morphology.remove_small_objects(mask, min_size=3, connectivity=1, in_place=False)
        #plt.imshow(clean_mask, origin="lower")
        #plt.show()
        print clean_mask.dtype
        para[:,~clean_mask] = np.nan

    # write the "cleaned" parameters
    newfits = fits.PrimaryHDU(data=para, header=hdr_para)
    newfits.writeto(savename ,overwrite=True)

    return para

def make_SNR_map(paraname, savename):
    return mvf.get_SNR(paraname, savename = savename, rms = 0.15, n_comp = 2)

#=======================================================================================================================

def quickFix(regname = "NGC1333"):
    # my quick way to generate the correct 2 component model cube and calculate chisq values
    # the original pipeline that I run at the UVic server had some problems
    from astropy.wcs import WCS
    from spectral_cube import SpectralCube

    workDir = "/Users/mcychen/Documents/Data/GAS_NH3/DR1_rebase3/{0}".format(regname)

    mod1name = "{0}/{1}_NH3_11_DR1_rebase3_sepVModel_v0_.fits".format(workDir,regname)
    mod2name = "{0}/{1}_NH3_11_DR1_rebase3_sepVModel_v1_.fits".format(workDir,regname)
    obsname = "{0}/{1}_NH3_11_DR1_rebase3_trim.fits".format(workDir,regname)

    chiname = "{0}/{1}_NH3_11_DR1_rebase3_trim_chisq.fits".format(workDir,regname)
    modname = "{0}/{1}_NH3_11_DR1_rebase3_trim_model.fits".format(workDir,regname)

    cube_m1, hdr_m1 = fits.getdata(mod1name, header=True)
    cube_m2, hdr_m2 = fits.getdata(mod2name, header=True)
    cube_o, hdr_o = fits.getdata(obsname, header=True)

    # the total model spectrum
    cube_m = np.nansum([cube_m1, cube_m2], axis = 0)

    spcube_o = SpectralCube(cube_o, WCS(hdr_o), header=hdr_o)
    chisq = mvf.get_chisq(spcube_o, cube_m, expand=20)

    chisqfile = fits.PrimaryHDU(data=chisq, header=WCS(hdr_o).celestial.to_header())
    chisqfile.writeto(chiname, overwrite=True)

    modfile = fits.PrimaryHDU(data=cube_m, header=hdr_o)
    modfile.writeto(modname, overwrite=True)


