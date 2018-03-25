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

    def __init__(self, region_name, root='base_all_rebase3', rootPara = None):

        self.region = region_name
        self.rootDir = "/Users/mcychen/Documents/Data/GAS_NH3"
        self.cubeDir = "{0}/{1}/{2}".format(self.rootDir, 'DRMC_rebase3', self.region)
        self.paraDir = "{0}/{1}/{2}/{3}".format(self.rootDir, 'DRMC_paraMaps', self.region, 'paraMaps_MChen')
        #self.modelDir = "{0}/{1}/multi_v_models_MChen".format(self.cubeDir, self.region)
        self.cleanParaDir = "{0}/clean_maps".format(self.paraDir)
        self.cubeFigDir = "{0}/{1}".format(self.cubeDir, "figures")

        # create directories if they don't exist
        make_dir(self.cubeFigDir)

        self.root = root
        if rootPara is None:
            self.rootPara = self.root

        self.OneOneFile = '{0}/{1}_NH3_11_{2}.fits'.format(self.cubeDir, self.region, self.rootPara)

        '''
        # defining paths to imaged data
        self.RMSFile = '{2}/{0}/{0}_NH3_11_{1}_rms.fits'.format(self.region, self.root, self.cubeDir)
        self.SingVParaFile = '{2}/{0}/{0}_parameter_maps_{1}.fits'.format(self.region, self.rootPara, self.cubeDir)
        '''


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

        ParaFile = '{2}/{0}_{3}vcomp_parameter_maps_{1}.fits'.format(self.region, self.rootPara, self.paraDir,
                                                                             n_comp)

        SNRFile = '{2}/{0}_{3}vcomp_SNR_eachV_{1}.fits'.format(self.region, self.rootPara, self.paraDir, n_comp)

        # output file paths

        self.CleanParaFile_2comp = '{2}/{0}_{3}vcomp_parameter_maps_{1}_clean.fits'.format(self.region, self.rootPara,
                                                                                     self.cleanParaDir, n_comp)

        aiccpath = "{0}/{1}_NH3_11_{2}v{3}comp_aicc.fits".format(self.cleanParaDir, self.region, n_comp, n_comp_compare)


        clean_map.clean(ParaFile, self.CleanParaFile_2comp, snrname = SNRFile, one_v_map = self.CleanParaFile_1comp,
                        aic_maps = aiccpath, fill_plane1 = False, mergesim = False, rm_sml_obj = True,
                        sort_method = "None", snr_min = 3.0)



    def clean_map_1comp(self):
        n_comp = 1
        ParaFile1 = '{2}/{0}_{3}vcomp_parameter_maps_{1}.fits'.format(self.region, self.rootPara, self.paraDir, n_comp)
        SNRFile = '{2}/{0}_{3}vcomp_SNR_eachV_{1}.fits'.format(self.region, self.rootPara, self.paraDir, n_comp)

        self.CleanParaFile_1comp = '{2}/{0}_{3}vcomp_parameter_maps_{1}_clean.fits'.format(self.region, self.rootPara,
                                                                                     self.cleanParaDir, n_comp)

        clean_map.clean_onecomp(ParaFile1, self.CleanParaFile_1comp, snrname = SNRFile, rm_sml_obj = True, snr_min = 3.0)


    def deblend_2comp(self, vmin, vmax, res_boost = 1.0):

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

        # make a deblended cube at the native resolution
        if res_boost == 1.0:
            self.DeblendFile = "{0}/{1}_NH3_11_{2}_2vcomp_deblended_NyqFixSig.fits".format(self.deblendDir, self.region,
                                                                                           self.root)
        else:
            self.DeblendFile = "{0}/{1}_NH3_11_{2}_2vcomp_deblended_NyqFixSig_{3}xRes.fits".format(self.deblendDir,
                                                                                                   self.region,
                                                                                                   self.root,
                                                                                                   int(res_boost))

        deblend_cube.deblend_cube(ParaFile, self.OneOneFile, self.DeblendFile, vmin=vmin, vmax=vmax, T_bg=0.0,
                                  sigv_fixed=sigv, f_spcsamp=res_boost)

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
            print "goooood"
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
# one time function


def clean_reg(reg = 'HC2'):
    region = Region(reg)
    region.clean_map()
    return None

def deblend(reg = 'HC2', vmin=4, vmax=7):
    region = Region(reg)
    region.deblend_2comp(vmin, vmax)

def check_fit():

    if False:
        reg = 'L1448'
        region = Region(reg)
        yxList = [(105,86),
            (116,99),
            (118,110),
            (125,106),
            (123,125),
            (120,140)]
        region.exam_spec_fits(yxList, n_comp=1)


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


    if True:
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

