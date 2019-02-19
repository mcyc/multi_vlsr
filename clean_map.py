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
    # a quick test run to see what the AIC maps look like
    baseDir = "/Users/mcychen/Documents/Data/GAS_NH3/DR1_rebase3/NGC1333"
    aiccname = "{0}/two_v_comp/NGC1333_2v1comp_aicc.fits".format(baseDir)
    #aiccname = "{0}/two_v_comp/NGC1333_2v1comp_chisq.fits".format(baseDir)

    aicc = fits.getdata(aiccname)
    aicc[aicc == 0.0] = np.nan

    #mask = aicc[0] < aicc[1]
    # True where the second model is 2 times as probable as the first model to minimize the information loss
    #mask = np.exp(aicc[0] - aicc[1])/2.0 > 1.0
    mask = aicc[0] > aicc[1]

    #plt.imshow(aicc[0]-aicc[1], origin="lower", interpolation="nearest")
    #plt.imshow(aicc[0]/aicc[1], origin="lower", interpolation="nearest")

    plt.imshow(mask, origin="lower", interpolation="nearest")


    if False:
        map = np.exp(aicc[0] - aicc[1])/2.0
        plt.imshow(np.log10(map), origin="lower", interpolation="nearest", clim=(0.0, 3))
        plt.colorbar()

    if False:
        yo = aicc[1].copy()
        yo[~mask] = np.nan
        yo[yo == 0] = np.nan

        plt.imshow(yo, origin="lower", interpolation="nearest")

    if False:
        ratio = (aicc[0]/aicc[1]).ravel()
        ratio = ratio[~np.isnan(ratio)]
        plt.hist(ratio, bins=100, cumulative=True, normed=True)

    plt.show()


def clean_reg(reg = "NGC1333"):
    # clean the 2 component fit of a region using the defaults

    baseDir = "/Users/mcychen/Documents/Data/GAS_NH3"
    workDir = "{0}/DR1_paraMaps/{1}/two_v_comp".format(baseDir, reg)
    if not os.path.exists(workDir):
        os.makedirs(workDir)

    oriname = "{0}/{1}_2vcomp_parameter_maps_DR1_rebase3.fits".format(workDir, reg)
    newname = "{0}/{1}_2vcomp_SNR_eachV_DR1_rebase3_clean.fits".format(workDir, reg)
    #newname = "{0}/{1}_2vcomp_DR1_rebase3_clean.fits".format(workDir, reg)
    snrname = "{0}/{1}_2vcomp_SNR_eachV_DR1_rebase3.fits".format(workDir, reg)

    aicname = "{0}/DR1_rebase3/{1}/two_v_comp/{1}_2v1comp_aicc.fits".format(baseDir, reg)
    one_v_name = "{0}/DR1_paraMaps/{1}/one_v_comp/{1}_1vcomp_parameter_maps_DR1_rebase3_clean.fits".format(baseDir, reg)

    clean(oriname, newname, snrname = snrname, one_v_map = one_v_name, aic_maps = aicname,
          fill_plane1 = False, mergesim = False, rm_sml_obj = True, sort_method = "None")

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








def clean_2comp(readname, savename, one_v_map=None, aic_maps=None, fill_plane1=False, rm_sml_obj=True, sort_method=None):
    '''
    Cleaning a two velocity component parameter map
    '''

    # the maximum vlsr error allowed (km/s)
    #vErr_Thres = 0.1
    vErr_Thres = 0.5

    para, hdr_para = fits.getdata(readname, header = True)

    # remove the zero values
    para[para == 0] = np.nan

    v1, sig1, tex1, tau1 = para[:4]
    v2, sig2, tex2, tau2 = para[4:8]
    v1_err, sig1_err, tex1_err, tau1_err = para[8:12]
    v2_err, sig2_err, tex2_err, tau2_err = para[12:]

    # remove the zero values in vlsr
    #mask_all = np.logical_or(v1 == 0, v2 == 0)

    # use the 1-component model whenever the 1-comp model is better based on AIC
    if not one_v_map is None:
        para_1v, hdr_1v = fits.getdata(one_v_map, header = True)
        para_1v[para_1v == 0.0] = np.nan

        # remove components that have vlsr error larger than the threshold
        eMask = v1_err > vErr_Thres
        para[:4, eMask] = np.nan
        para[8:12, eMask] = np.nan

        eMask = v2_err > vErr_Thres
        para[4:8, eMask] = np.nan
        para[12:, eMask] = np.nan

        if not aic_maps is None:
            aic, hdr_aic = fits.getdata(aic_maps, header = True)
            # True where the single component fit is better than the two component fit

            # likelyhood factor thershold
            lk_factor = 100.0

            # where the second model is lk_factor times as probable as the first model to minimize the information loss
            likelyhood = np.exp((aic[0] - aic[1])/2.0)
            mask_sg = likelyhood > lk_factor

            # replace the first components with the single component values, and set the second component to zero
            para[:4, ~mask_sg] = para_1v[:4, ~mask_sg]
            para[4:8, ~mask_sg] = np.nan

            para[8:12, ~mask_sg] = para_1v[4:, ~mask_sg]
            para[12:, ~mask_sg] = np.nan


        else:
            print "warning: no aic map is provided."


        # replace pixels that does not have two components after cleaning with that from a single component model
        mask = np.logical_and(~np.isnan(para[0]), ~np.isnan(para[4]))

        para[:4, ~mask] = para_1v[:4, ~mask]
        para[4:8, ~mask] = np.nan

        para[8:12, ~mask] = para_1v[4:, ~mask]
        para[12:, ~mask] = np.nan



    if sort_method is not None:
        # sort the front and rear components by the specified method
        if sort_method == "v_error":
            swap = v2_err > v1_err

        elif sort_method == "linewidth":
            # place components with larger linewidths into the second componet
            # (i.e., what is more likely the ambient gas)
            #swap = sig1 > sig2+np.hypot(sig1_err, sig1_err)
            swap = sig1 > sig2

        else:
            print "[WARNING]: /'{0}/' is not recognized, no sorting is performed".format(sort_method)
            swap = np.logical_and(np.isnan(v1), np.isnan(v2))

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


    if rm_sml_obj:
        # remove islated pixels of a certain size
        mask = ~np.isnan(para)
        mask = mask.sum(axis=0)
        mask = mask/mask
        mask = mask.astype(bool)
        #plt.imshow(mask, origin="lower")
        #plt.show()
        clean_mask = morphology.remove_small_objects(mask, min_size=5, connectivity=1, in_place=False)
        #plt.imshow(clean_mask, origin="lower")
        #plt.show()

        para[:,~clean_mask] = np.nan

    # write the "cleaned" parameters
    newfits = fits.PrimaryHDU(data=para, header=hdr_para)
    newfits.writeto(savename ,overwrite=True)

    return para








# this is an older implementation. use clean_2comp() for the updated version
def clean(readname, savename, snrname = None, one_v_map = None, aic_maps = None,
          fill_plane1 = False, mergesim = False, rm_sml_obj = True, sort_method = "None", snr_min = 3.0):
    '''
    Cleaning a two velocity component parameter map
    A good recipe so far:
    1. SNR culling
    2. Place larger-line width component on the second plane
    3. Merge pixels with similar velocity and linewdiths into one, and place it on the 1st plane
    '''

    para, hdr_para = fits.getdata(readname, header = True)

    v1, sig1, tex1, tau1 = para[:4]
    v2, sig2, tex2, tau2 = para[4:8]
    v1_err, sig1_err, tex1_err, tau1_err = para[8:12]
    v2_err, sig2_err, tex2_err, tau2_err = para[12:]

    # remove the zero values in vlsr
    mask_all = np.logical_or(v1 == 0, v2 == 0)

    # remove pixels with low SNR
    if snrname is not None:
        snr, hdr_snr = fits.getdata(snrname, header = True)

        '''
        mask_1 = snr[0] < snr_min
        mask_2 = snr[1] < snr_min
        # mask out first velocity component
        para[:4, mask_1], para[8:12, mask_1] = np.nan, np.nan
        # mask out second velocity component
        para[4:8, mask_2], para[12:, mask_2] = np.nan, np.nan
        '''
        # SNR culling on individual pixels can often result in very choppy map. So for now, I'll going to do the SNR
        # culling on the total emission combined, and let AIC decided whether the extra parameters are justified

        mask = snr[0] + snr[1] < snr_min
        # mask out first velocity component
        para[:, mask] = np.nan

        if not one_v_map is None:
            # remove a pixel if individual component does not meet the following SNR criteria
            # the removed pixel will be replaced by the single component model provided
            snr_min_onecomp = 1.0
            mask = np.logical_or(snr[0] < snr_min_onecomp, snr[1] < snr_min_onecomp)
            para[:, mask] = np.nan


    '''
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

    # remove pixels where the vlsr error is above a certain threshhold
    evlsr_thr = 0.08 # about the GAS spectral channel width
    mask_1 = np.logical_or(mask_1, v1_err >= evlsr_thr)
    mask_2 = np.logical_or(mask_2, v2_err >= evlsr_thr)

    # mask out all maps
    para[:, mask_all] = np.nan

    # mask out first velocity component
    para[:4, mask_1], para[8:12, mask_1] = np.nan, np.nan

    # mask out second velocity component
    para[4:8, mask_2], para[12:, mask_2] = np.nan, np.nan
    '''

    # use the 1-component model whenever the 1-comp model is better based on AIC
    if not one_v_map is None:
        para_1v, hdr_1v = fits.getdata(one_v_map, header = True)
        para_1v[para_1v == 0.0] = np.nan

        '''
        # leave pixels where both v1 and v2 are NaN untouched
        mask_all = np.logical_and(v1 == np.nan, v2 == np.nan)
        para_1v[:,mask_all] = np.nan
        '''

        if not aic_maps is None:
            aic, hdr_aic = fits.getdata(aic_maps, header = True)
            # True where the single component fit is better than the two component fit

            # likelyhood factor thershold
            lk_factor = 100.0

            #mask_sg = aic[0] < aic[1]

            # where the second model is lk_factor times as probable as the first model to minimize the information loss
            likelyhood = np.exp((aic[0] - aic[1])/2.0)
            mask_sg = likelyhood > lk_factor

            # replace the first components with the single component values, and set the second component to zero
            para[:4, ~mask_sg] = para_1v[:4, ~mask_sg]
            para[4:8, ~mask_sg] = np.nan

            para[8:12, ~mask_sg] = para_1v[4:, ~mask_sg]
            para[12:, ~mask_sg] = np.nan


        else:
            print "warning: no aic map is provided."

        # replace pixels that does not have two components after cleaning with that from a single component model
        mask = np.logical_and(~np.isnan(para[0]), ~np.isnan(para[4]))

        para[:4, ~mask] = para_1v[:4, ~mask]
        para[4:8, ~mask] = np.nan

        para[8:12, ~mask] = para_1v[4:, ~mask]
        para[12:, ~mask] = np.nan



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
    # lolololo

    '''
    #if SNRSort:
    if sort_method == "SNR":
        if snrname is not None:
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

    '''

    if rm_sml_obj:
        # remove islated pixels of a certain size
        mask = ~np.isnan(para)
        mask = mask.sum(axis=0)
        mask = mask/mask
        mask = mask.astype(bool)
        #plt.imshow(mask, origin="lower")
        #plt.show()
        clean_mask = morphology.remove_small_objects(mask, min_size=5, connectivity=1, in_place=False)
        #plt.imshow(clean_mask, origin="lower")
        #plt.show()

        para[:,~clean_mask] = np.nan

    # write the "cleaned" parameters
    newfits = fits.PrimaryHDU(data=para, header=hdr_para)
    newfits.writeto(savename ,overwrite=True)

    return para



def clean_1v():
    workDir = "/Users/mcychen/Documents/Data/GAS_NH3/DR1_paraMaps/NGC1333/one_v_comp"
    readname = "{0}/NGC1333_1vcomp_parameter_maps_DR1_rebase3.fits".format(workDir)
    savename = "{0}/NGC1333_1vcomp_parameter_maps_DR1_rebase3_clean.fits".format(workDir)
    clean_onecomp(readname, savename, snrname = None, rm_sml_obj = True)



def clean_onecomp(readname, savename, snrname = None, rm_sml_obj = True, snr_min = 3.0):
    '''
    Cleaning a single component parameter map
    A good recipe so far:
    1. SNR culling
    '''

    para, hdr_para = fits.getdata(readname, header = True)

    # replace all zero values with nans
    para[para == 0.0] = np.nan

    v1, sig1, tex1, tau1 = para[:4]
    v1_err, sig1_err, tex1_err, tau1_err = para[4:8]

    # remove pixels with low SNR
    if not snrname is None:
        snr, hdr_snr = fits.getdata(snrname, header = True)
        mask_1 = snr[0] < snr_min
        # mask out first velocity component
        para[:4, mask_1], para[4:8, mask_1] = np.nan, np.nan


    # remove pixels with line width that are outside the expected physical range
    sig_min = 0.05
    mask_1 = sig1 <= sig_min

    sig_max = 5.0           # all the hyperfines starts to blend together at this point
    mask_1 = np.logical_or(mask_1, sig1 >= sig_max)


    # remove pixels where the error in the line width is larger than linewidth itself
    mask_1 = np.logical_or(mask_1, sig1 <= sig1_err)

    # remove pixels where the vlsr error is above a certain threshhold
    evlsr_thr = 0.08 # about the GAS spectral channel width
    mask_1 = np.logical_or(mask_1, v1_err >= evlsr_thr)

    # mask out first velocity component
    para[:4, mask_1], para[4:8, mask_1] = np.nan, np.nan


    if rm_sml_obj:
        # remove islated pixels of a certain size
        mask = ~np.isnan(para)
        mask = mask.sum(axis=0)
        mask = mask/mask
        mask = mask.astype(bool)
        clean_mask = morphology.remove_small_objects(mask, min_size=5, connectivity=1, in_place=False)
        #print clean_mask.dtype
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


