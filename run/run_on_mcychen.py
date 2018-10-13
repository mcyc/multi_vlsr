__author__ = 'mcychen'

import sys, os, errno, time
# add the parent directory to the paths
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import multi_v_fit as mvf
import aic
import clean_map
import exam_fit
import deblend_cube
reload(clean_map)
reload(exam_fit)
reload(deblend_cube)


class Region(object):
    '''
    A Class that contains all the relevant file path and run the pipeline fitting and analysis
    '''

    def __init__(self, region_name, root='all_rebase3', rootPara = None, linename = "oneone"):

        self.region = region_name
        self.rootDir = "/Users/mcychen/Documents/Data/GAS_NH3"
        self.cubeDir = "{0}/{1}/{2}".format(self.rootDir, 'DRMC_rebase3', self.region)
        self.paraDir = "{0}/{1}/{2}/{3}".format(self.rootDir, 'DRMC_paraMaps', self.region, 'paraMaps_MChen')
        #self.modelDir = "{0}/{1}/multi_v_models_MChen".format(self.cubeDir, self.region)
        self.cleanParaDir = "{0}/clean_maps".format(self.paraDir)
        self.cubeFigDir = "{0}/{1}".format(self.cubeDir, "figures")

        # create directories if they don't exist
        make_dir(self.cubeFigDir)
        make_dir(self.cleanParaDir)

        self.root = root
        if rootPara is None:
            self.rootPara = self.root

        self.linename = linename
        if linename == "oneone":
            self.line_root = "11"
        elif linename == "twotwo":
            self.line_root = "22"
        else:
            self.line_root = "bad"
            print "[ERROR]: NH3 lines beyond twotwo has yet to be implemented"
        self.OneOneFile = '{0}/{1}_NH3_{2}_{3}.fits'.format(self.cubeDir, self.region, self.line_root, self.rootPara)


    def clean_map(self):
        # currently wraps around cleaning methods for one-component fits and two-component fits
        self.clean_map_1comp()
        self.clean_map_2comp()


    def clean_map_2comp(self):
        if not hasattr(self, 'CleanParaFile_1comp'):
            self.clean_map_1comp()

        # currently only available for cleaning 2 component models
        n_comp = 2
        n_comp_compare = 1

        ParaFile = '{2}/{0}_NH3_{4}_{3}vcomp_parameter_maps_{1}.fits'.format(self.region, self.rootPara, self.paraDir,
                                                                             n_comp, self.line_root)

        SNRFile = '{2}/{0}_NH3_{4}_{3}vcomp_SNR_eachV_{1}.fits'.format(self.region, self.rootPara, self.paraDir, n_comp,
                                                                       self.line_root)

        # output file paths
        self.CleanParaFile_2comp = '{2}/{0}_NH3_{4}_{3}vcomp_parameter_maps_{1}_clean.fits'.format(self.region,
                                                                                                   self.rootPara,
                                                                                                   self.cleanParaDir,
                                                                                                   n_comp,
                                                                                                   self.line_root)

        aiccpath = "{0}/{1}_NH3_{4}_{2}v{3}comp_aicc.fits".format(self.cleanParaDir, self.region, n_comp,
                                                                  n_comp_compare, self.line_root)

        '''
        clean_map.clean(ParaFile, self.CleanParaFile_2comp, snrname = SNRFile, one_v_map = self.CleanParaFile_1comp,
                        aic_maps = aiccpath, fill_plane1 = False, mergesim = False, rm_sml_obj = True,
                        sort_method = "None", snr_min = 3.0)

        clean_map.clean(ParaFile, self.CleanParaFile_2comp, snrname = None, one_v_map = self.CleanParaFile_1comp,
                        aic_maps = aiccpath, fill_plane1 = False, mergesim = False, rm_sml_obj = True,
                        sort_method = "None", snr_min = 3.0)
        '''

        clean_map.clean_2comp(ParaFile, self.CleanParaFile_2comp, one_v_map=self.CleanParaFile_1comp,
                              aic_maps=aiccpath, fill_plane1=True, rm_sml_obj=True, sort_method="linewidth")



    def clean_map_1comp(self):
        n_comp = 1

        ParaFile1 = '{2}/{0}_NH3_{4}_{3}vcomp_parameter_maps_{1}.fits'.format(self.region, self.rootPara, self.paraDir,
                                                                              n_comp, self.line_root)

        SNRFile = '{2}/{0}_NH3_{4}_{3}vcomp_SNR_eachV_{1}.fits'.format(self.region, self.rootPara, self.paraDir, n_comp,
                                                                       self.line_root)

        self.CleanParaFile_1comp = '{2}/{0}_NH3_{4}_{3}vcomp_parameter_maps_{1}_clean.fits'.format(self.region,
                                                                                                   self.rootPara,
                                                                                                   self.cleanParaDir,
                                                                                                   n_comp, self.line_root)

        #clean_map.clean_onecomp(ParaFile1, self.CleanParaFile_1comp, snrname = SNRFile, rm_sml_obj = True, snr_min = 3.0)
        clean_map.clean_onecomp(ParaFile1, self.CleanParaFile_1comp, snrname = None, rm_sml_obj = True, snr_min = 3.0)


    def deblend_2comp(self, vmin, vmax, res_boost = 1.0, fixsigma=True):

        if not hasattr(self, 'CleanParaFile_2comp'):
            self.clean_map_2comp()

        if not hasattr(self, 'deblendDir'):
            self.deblendDir = "{0}/{1}".format(self.cubeDir, "deblended")
            make_dir(self.deblendDir)

        ParaFile = self.CleanParaFile_2comp

        # the velocity resolution of each channel
        v_rez = 0.0724
        # the number of pixels to sample a FWHM beam at nyquist limit
        n_pix_nyq_samp = 3 # conforming to GAS spatial sampling
        # conversion factor between FWHM and sigma
        fwhm_per_sig = 2.355

        # fixed the linewidth at the narrowest allowed by the nyquist sampling
        # the following value is ~0.0769 km/s if n_pix_nyq_samp = 2.5
        sigv = v_rez*n_pix_nyq_samp/fwhm_per_sig
        sigv = sigv/res_boost

        if fixsigma:
            root_sigma = "_NyqFixSig"
        else:
            root_sigma = ""

        # make a deblended cube at the native resolution
        if res_boost == 1.0:
            self.DeblendFile = "{0}/{1}_NH3_11_{2}_2vcomp_deblended{3}.fits".format(self.deblendDir, self.region,
                                                                                           root_sigma, self.root)
        else:
            self.DeblendFile = "{0}/{1}_NH3_11_{2}_2vcomp_deblended{3}_{4}xRes.fits".format(self.deblendDir,
                                                                                            self.region, self.root,
                                                                                            root_sigma, int(res_boost))

        if fixsigma:
            deblend_cube.deblend_cube(ParaFile, self.OneOneFile, self.DeblendFile, vmin=vmin, vmax=vmax, T_bg=0.0,
                                  sigv_fixed=sigv, f_spcsamp=res_boost)
        else:
            deblend_cube.deblend_cube(ParaFile, self.OneOneFile, self.DeblendFile, vmin=vmin, vmax=vmax, T_bg=0.0,
                                      f_spcsamp=res_boost)


        #outfile = "{0}/{1}_NH3_11_{2}_2vcomp_deblended.fits".format(self.deblendDir, self.region, self.root)

        #ParaFile = "/Users/mcychen/Documents/Data/GAS_NH3/DRMC_paraMaps/HC2/paraMaps_MChen/HC2_2vcomp_parameter_maps_base_all_rebase3.fits"
        #deblend_cube.deblend_cube(ParaFile, self.OneOneFile, outfile, vmin=vmin, vmax=vmax, T_bg=0.0)


    def exam_spec_fits(self, yxList, n_comp, vZoomLims = None, useCleanMap = False):

        if useCleanMap:
            # note: only works for 2-components rigth now!!!
            if n_comp !=2:
                print "[ERROR]: useCleanMap only works with 2 components maps at the moment"
                return None

            if not hasattr(self, 'CleanParaFile_2comp'):
                self.clean_map_2comp()

            # Note to self: I need to calculate chi-squared values on the spot for this one instead of relying on a
            # pre-existing map
            parapath = self.CleanParaFile_2comp
            saveSpecPath = '{0}/{1}_spec_{2}comp_clean.pdf'.format(self.cubeFigDir, self.region, n_comp)

        else:
            parapath = '{2}/{0}_{3}vcomp_parameter_maps_{1}.fits'.format(self.region, self.rootPara, self.paraDir, n_comp)
            saveSpecPath = '{0}/{1}_spec_{2}comp.pdf'.format(self.cubeFigDir, self.region, n_comp)

        chipath = '{0}/{1}_{2}vcomp_chisq_{3}.fits'.format(self.paraDir, self.region, n_comp, self.rootPara)

        # plot spectral fits

        exam_fit.plotMultiSpec(parapath, n_comp, self.OneOneFile, chipath, yxList, saveSpecPath, showSpec = False,
                               vZoomLims =vZoomLims)

        # plot positions of the spectrum
        # note: this may have to change as the pipeline moves on
        self.rootDir = "/Users/mcychen/Documents/Data/GAS_NH3"
        mom0Dir = "{0}/{1}".format(self.rootDir, "DR2_NH3_11_moments/mom0_maps")
        mom0path = '{0}/{1}_NH3_11_base_all_rebase3_mom0.fits'.format(mom0Dir, self.region)
        saveMappath = '{0}/{1}_locOfSpec_{2}comp.pdf'.format(self.cubeFigDir, self.region, n_comp)
        exam_fit.mapPositions(mom0path, yxList, saveMappath, showMap = False,
                              cBarLab=r"Integrated Intensity (K km s$^{-1}$)")


#=======================================================================================================================
# some quick utilities functions

def make_dir(dirpath):
    # create directories if they don't exist
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

########################################################################################################################
# user inputs

def clean_reg(reg = 'L1448'):
    region = Region(reg, linename = "oneone")
    region.clean_map()
    #region = Region(reg, linename = "twotwo")
    #region.clean_map()
    return None

def deblend(reg = 'HC2', vmin=4, vmax=7, **kwargs):
    region = Region(reg)
    region.deblend_2comp(vmin, vmax, *kwargs)

def clean_DR1_reg(reg='NGC1333', linename = "oneone"):
    region = Region(reg, linename=linename, root='DR1_rebase3')
    region.cubeDir = "{0}/{1}/{2}".format(region.rootDir, 'DR1_rebase3', region.region)
    region.paraDir = "{0}/{1}/{2}/{3}".format(region.rootDir, 'DR1_paraMaps', region.region, 'paraMaps_MChen')
    region.cleanParaDir = "{0}/clean_maps".format(region.paraDir)
    region.cubeFigDir = "{0}/{1}".format(region.cubeDir, "figures")
    region.clean_map()

def deblend_DR1(reg='NGC1333', vmin=5.2, vmax=9.3, linename = "oneone", **kwargs):
    region = Region(reg, linename=linename, root='DR1_rebase3')
    region.cubeDir = "{0}/{1}/{2}".format(region.rootDir, 'DR1_rebase3', region.region)
    region.OneOneFile = '{0}/{1}_NH3_{2}_{3}.fits'.format(region.cubeDir, region.region, region.line_root, region.rootPara)
    print region.cubeDir
    #return None
    #self.cubeDir = "{0}/{1}/{2}".format(self.rootDir, 'DRMC_rebase3', self.region)

    region.paraDir = "{0}/{1}/{2}/{3}".format(region.rootDir, 'DR1_paraMaps', region.region, 'paraMaps_MChen')
    region.cleanParaDir = "{0}/clean_maps".format(region.paraDir)
    region.cubeFigDir = "{0}/{1}".format(region.cubeDir, "figures")
    region.deblend_2comp(vmin, vmax, **kwargs)

########################################################################################################################
# one time functions

def check_fit():

    if True:
        reg = 'L1448'
        region = Region(reg)
        yxList = [(105,86),
            (116,99),
            (118,110),
            (125,106),
            (123,125),
            (120,140)]
        region.OneOneFile = '{0}/{1}_NH3_11_{2}.fits'.format(region.cubeDir, region.region, 'all_rebase_multi')
        region.exam_spec_fits(yxList, n_comp=2, vZoomLims=(-6,15))


    if False:
        reg = 'OrionB_NGC2023-2024'
        region = Region(reg)
        '''
        region.OneOneFile = "/Users/mcychen/Documents/Data/GAS_NH3/DRMC_rebase3/OrionB_NGC2023-2024/" \
                            "OrionB_NGC2023-2024_NH3_11_all_rebase_multi.fits"
        '''
        region.OneOneFile = "/Users/mcychen/Documents/Data/GAS_NH3/DRMC_rebase3/OrionB_NGC2023-2024/" \
                            "OrionB_NGC2023-2024_NH3_11_all_rebase3.fits"
        yxList = [(372,104),
            (342,122),
            (287,68),
            (132,72),
            (126,98),
            (85,92)]
        region.exam_spec_fits(yxList, n_comp=1)


    if False:
        reg = 'HC2'
        region = Region(reg)
        '''
        yxList = [(249,124),
            (219,107),
            (189,75),
            (142,40),
            (215,254),
            (203,258)]
        '''
        yxList = [(249,124),
            (219,107),
            (205,83),
            (142,40),
            (215,254),
            (203,258)]
        region.exam_spec_fits(yxList, n_comp=2, vZoomLims=(-5,17), useCleanMap=False)

    return None


def ccc():
    # quickly check the quality of the debelended cube
    reg = 'HC2'
    region = Region(reg)
    #region.clean_map()
    #region.deblend_2comp(4, 7)
    yxList = [(206,84),
        (205,83),
        (206,83),
        (205,84),
        (207,85),
        (207,84)]

    '''
    yxList = [(249,124),
        (219,107),
        (189,75),
        (142,40),
        (215,254),
        (203,258)]
    '''
    #region.exam_spec_fits(yxList, n_comp=1, vZoomLims=(-5,17), useCleanMap=False)
    region.exam_spec_fits(yxList, n_comp=2, vZoomLims=(-5,17), useCleanMap=False)
    #region.exam_spec_fits(yxList, n_comp=2, vZoomLims=(-5,17), useCleanMap=True)


def ckcl():
    # check the cleaning technique
    reg = 'L1448'
    region = Region(reg)
    '''
    yxList = [(103,84),
            (116,99),
            (118,110),
            (125,106),
            (125,122),
            (117,139)]
    '''
    yxList = [(123,107),
            (123,106),
            (123,105),
            (102,84),
            (102,85),
            (102,86)]

    region.OneOneFile = '{0}/{1}_NH3_11_{2}.fits'.format(region.cubeDir, region.region, 'all_rebase_multi')
    vZoomLims = (-6,15)
    region.exam_spec_fits(yxList, n_comp=1, vZoomLims=vZoomLims, useCleanMap=False)
    #region.exam_spec_fits(yxList, n_comp=2, vZoomLims=vZoomLims, useCleanMap=False)
    #region.exam_spec_fits(yxList, n_comp=2, vZoomLims=vZoomLims, useCleanMap=True)


def comp_11n22(region = "NGC1333", root='DR1_rebase3'):
    #region = "L1448", root='base_all_rebase3'

    import astropy.io.fits as fits
    import matplotlib.pyplot as plt
    import numpy as np

    baseDir = "/Users/mcychen/Documents/Data/GAS_NH3/DRMC_paraMaps/{0}/paraMaps_MChen/clean_maps".format(region)
    f1 = "{0}/{1}_NH3_{2}_2vcomp_parameter_maps_{3}_clean.fits".format(baseDir, region, "11", root)
    f2 = "{0}/{1}_NH3_{2}_2vcomp_parameter_maps_{3}_clean.fits".format(baseDir, region, "22", root)

    '''
    # uncleaned maps
    baseDir = "/Users/mcychen/Documents/Data/GAS_NH3/DRMC_paraMaps/{0}/paraMaps_MChen".format(region)
    f1 = "{0}/{1}_NH3_{2}_2vcomp_parameter_maps_{3}.fits".format(baseDir, region, "11", root)
    f2 = "{0}/{1}_NH3_{2}_2vcomp_parameter_maps_{3}.fits".format(baseDir, region, "22", root)
    '''

    data11 = fits.getdata(f1)
    data22 = fits.getdata(f2)

    data11[data11 == 0.0] = np.nan
    data22[data22 == 0.0] = np.nan

    # sort the components by their similarity
    swap = np.abs(data11[0]-data22[0]) > np.abs(data11[0]-data22[4])
    data11[0:4,swap], data11[4:8,swap] = data11[4:8,swap], data11[0:4,swap]
    data11[8:12,swap], data11[12:,swap] = data11[12:,swap], data11[8:12,swap]

    f1 = "{0}/{1}_NH3_{2}_1vcomp_parameter_maps_{3}_clean.fits".format(baseDir, region, "11", root)
    data11_1v = fits.getdata(f1)
    data11_1v[data11_1v == 0.0] = np.nan


    if False:
        mask = np.all(np.isfinite(data11), axis=0)

        vdiff = data11[0]-data22[0]
        verr = np.hypot(data11[8], data22[8])
        #mask = np.logical_and(mask, verr > 0.07)

        #plt.errorbar(data11[0,mask], data22[0,mask], xerr=data11[8,mask], yerr=data22[8,mask], fmt="o", ms=1, mfc='k', mec='k')
        #plt.errorbar(data11[1,mask], vdiff[mask], yerr=verr[mask], fmt="o", ms=1, mfc='k', mec='k')
        #plt.errorbar(verr[mask], vdiff[mask], yerr=verr[mask], fmt="o", ms=1, mfc='k', mec='k')
        #plt.axhline(y=0.0, color='0.5', linestyle='-')
        plt.scatter(data11[1,mask], data22[1,mask], s=3)
        plt.scatter(data11[1,~mask], data22[1,~mask], s=3)


    if True:
        # compare the 11 and 22 vlsr in with respect to the fitted error
        diff1 = (data11[0]-data22[0])/np.hypot(data11[8], data22[8])
        diff2 = (data11[4]-data22[4])/np.hypot(data11[12], data22[12])
        #diff1 = np.hypot(data11[8], data22[8])
        #diff2 = np.hypot(data11[12], data22[12])
        # mask where both components exist
        mask = np.all(np.isfinite(data11), axis=0)

        plt.clf()
        range = (-10,10)
        plt.hist(diff1[~mask], 50, range=range, histtype = "stepfilled", color="0.85")
        plt.hist(diff1[mask], 50, range=range, histtype = "step")
        plt.hist(diff2[mask], 50, range=range, histtype = "step")

        # regular histogram
        plt.xlabel(r"$\Delta$v$_{lsr}/\epsilon_{v_{lsr}}$")
        plt.ylabel("pixel counts")
        plt.legend(["the only component", "front component", "rear component"], frameon=False)
        plt.title(r"1,1 vs. 2,2 fits for NH$_3$ in {0}".format(region))
        plt.savefig("{0}/figures/{1}_11vs22_vlsrRel2Error_histo.pdf".format(baseDir,region))

    if False:
        # compare the 11 and 22 vlsr in with respect to the fitted error
        diff1 = (data11[1]-data22[1])/np.hypot(data11[9], data22[9])
        diff2 = (data11[5]-data22[5])/np.hypot(data11[13], data22[13])
        #diff1 = np.hypot(data11[8], data22[8])
        #diff2 = np.hypot(data11[12], data22[12])
        # mask where both components exist
        mask = np.all(np.isfinite(data11), axis=0)

        plt.clf()
        range = (-10,10)
        plt.hist(diff1[~mask], 50, range=range, histtype = "stepfilled", color="0.85")
        plt.hist(diff1[mask], 50, range=range, histtype = "step")
        plt.hist(diff2[mask], 50, range=range, histtype = "step")

        # regular histogram
        plt.xlabel(r"$\Delta\sigma_{v}/\epsilon_{\sigma_v}$")
        plt.ylabel("pixel counts")
        plt.legend(["the only component", "front component", "rear component"], frameon=False)
        plt.title(r"1,1 vs. 2,2 fits for NH$_3$ in {0}".format(region))
        plt.savefig("{0}/figures/{1}_11vs22_sigmaRel2Error_histo.pdf".format(baseDir,region))


    if False:
        # compare the 11 and 22 vlsr
        diff1 = data11[0]-data22[0]
        diff2 = data11[4]-data22[4]
        # mask where both components exist
        mask = np.all(np.isfinite(data11), axis=0)

        plt.clf()
        plt.hist(diff1[~mask], 50, range=(-0.5,0.5), histtype = "stepfilled", color="0.85")
        plt.hist(diff1[mask], 50, range=(-0.5,0.5), histtype = "step")
        plt.hist(diff2[mask], 50, range=(-0.5,0.5), histtype = "step")

        if False:
            mask = np.logical_and(mask, np.abs(data11[0]-data11[4])> 0.2)
            plt.hist(diff1[mask], 50, range=(-0.5,0.5), histtype = "step")

        # regular histogram
        plt.xlabel(r"$\Delta$v$_{lsr}$ (km s$^{-1}$)")
        plt.ylabel("pixel counts")
        plt.legend(["the only component", "front component", "rear component"], frameon=False)
        plt.title(r"1,1 vs. 2,2 fits for NH$_3$ in {0}".format(region))
        plt.savefig("{0}/figures/{1}_11vs22_vlsr_histo.pdf".format(baseDir,region))



    if False:
        # compare the 11 and 22 sigma
        diff1 = data11[1]-data22[1]
        diff2 = data11[5]-data22[5]
        # mask where both components exist
        mask = np.all(np.isfinite(data11), axis=0)
        plt.clf()
        plt.hist(diff1[~mask], 50, range=(-0.5,0.5), histtype = "stepfilled", color="0.85")
        plt.hist(diff1[mask], 50, range=(-0.5,0.5), histtype = "step")
        plt.hist(diff2[mask], 50, range=(-0.5,0.5), histtype = "step")

        if False:
            mask = np.logical_and(mask, np.abs(data11[0]-data11[4])> 0.4)
            plt.hist(diff1[mask], 50, range=(-0.5,0.5), histtype = "step")

        plt.xlabel(r"$\Delta \sigma_{v}$ (km s$^{-1}$)")
        plt.ylabel("pixel counts")
        plt.legend(["the only component", "front component", "rear component"], frameon=False)
        plt.title(r"1,1 vs. 2,2 fits for NH$_3$ in {0}".format(region))
        plt.savefig("{0}/figures/{1}_11vs22_sigma_histo.pdf".format(baseDir,region))

    if False:
        mask = np.isfinite(data11[4])
        plt.hist((data11[0][mask]-data11_1v[0][mask]).ravel(), 50, range=(-0.5,0.5), histtype = "step")
        plt.hist(-1*(data11[4][mask]-data11_1v[0][mask]).ravel(), 50, range=(-0.5,0.5), histtype = "step")
        plt.xlabel(r"$\Delta$v$_{lsr}$ (km s$^{-1}$)")
        plt.ylabel("pixel counts")

    if False:
        # find the difference between the two components
        mask = np.isfinite(data11[4])
        diff3 = data11[0][mask] - data11[4][mask]
        plt.hist(np.abs(diff3), 50, range=(0,2), histtype = "step")
        plt.xlabel(r"$\Delta$v$_{lsr}$ (km s$^{-1}$)")
        plt.ylabel("pixel counts")

    if False:
        plt.scatter(data22[0], data11[4]-data22[4], s=3)
    plt.show()