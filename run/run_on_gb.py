__author__ = 'mcychen'

import sys, os, errno, time
# add the parent directory to the paths
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import multi_v_fit as mvf
import iterative_fit as itf
import aic
import clean_map
import time


#=======================================================================================================================
'''
This is a wrapper to run on the Green Bank server for the multi_vlsr package
'''
#=======================================================================================================================

class Region(object):
    '''
    A Class that contains all the relevant file path and run the pipeline fitting and analysis
    '''

    # base_all_rebase3 is no good root for the moment being (e.g., netagive baseline)
    def __init__(self, region_name, root='all_rebase3', rootPara = None, linename = "oneone"):

        self.region = region_name
        self.rootDir = "/lustre/pipeline/scratch/GAS"
        self.cubeDir = "{0}/images".format(self.rootDir)
        self.paraDir = "{0}/{1}/paraMaps_MChen".format(self.cubeDir, self.region)
        self.modelDir = "{0}/{1}/multi_v_models_MChen".format(self.cubeDir, self.region)
        self.cleanParaDir = "{0}/clean_maps".format(self.paraDir)

        self.linename = linename
        if linename == "oneone":
            self.line_root = "11"
        elif linename == "twotwo":
            self.line_root = "22"
        else:
            self.line_root = "bad"
            print "[ERROR]: NH3 lines beyond twotwo has yet to be implemented"

        # create directories if they don't exist

        make_dir(self.paraDir)
        make_dir(self.modelDir)
        make_dir(self.cleanParaDir)

        self.root = root
        if rootPara is None:
            self.rootPara = self.root

        # defining paths to imaged data
        self.OneOneFile = '{2}/{0}/{0}_NH3_{3}_{1}.fits'.format(self.region, self.root, self.cubeDir, self.line_root)
        self.RMSFile = '{2}/{0}/{0}_NH3_{3}_{1}_rms.fits'.format(self.region, self.root, self.cubeDir, self.line_root)
        self.SingVParaFile = '{2}/{0}/{0}_parameter_maps_{1}.fits'.format(self.region, self.rootPara, self.cubeDir)


    def start_timing(self, message = True):
        # timestamp the start a timing interval
        if message:
            print "...start timing..."
        self.start_time = time.time()


    def end_timing(self):
        # timestamp the end of a timing interval and print the time elapsed
        end = time.time()
        delta_time = int(end - self.start_time)
        print "time elapsed: {0}:{1}".format(delta_time/60, delta_time%60)


    def fit_cube(self, n_comp, multicore=8, snr_min=5.0, mask_function=None, iterfit=True, refOneCompCnv=False):

        self.start_timing()

        # defining paths to the imaged data
        # NOTE: given that there are multiple components, the following attributes are better implamented as
        # dictoinaries than just a single string
        self.NewParaFile = '{2}/{0}_NH3_{4}_{3}vcomp_parameter_maps_{1}.fits'.format(self.region, self.rootPara,
                                                                                     self.paraDir, n_comp, self.line_root)
        self.ModelFile = '{2}/{0}_NH3_{4}_{1}_{3}comp_model.fits'.format(self.region, self.root, self.modelDir,
                                                                        n_comp, self.line_root)
        self.ChisqFile = '{2}/{0}_NH3_{4}_{3}vcomp_chisq_{1}.fits'.format(self.region, self.rootPara, self.paraDir,
                                                                          n_comp, self.line_root)

        self.SNRFile = '{2}/{0}_{3}vcomp_SNR_eachV_{1}.fits'.format(self.region, self.rootPara, self.paraDir, n_comp)
        self.SepVModelFile = '{2}/{0}_NH3_{4}_{1}_{3}VModel.fits'.format(self.region, self.root, self.modelDir, n_comp,                                                                  self.line_root)

        if iterfit:
            # perform iternative fitting
            kwargs = {'ncomp':n_comp, 'paraname':self.NewParaFile, 'modname':self.ModelFile, 'chisqname':self.ChisqFile,
                      'guesses':None, 'errmap11name':None, 'multicore':multicore, 'snr_min':snr_min,
                      'mask_function':mask_function, 'linename':self.linename}

            if refOneCompCnv:
                singCompRef='{2}/{0}_NH3_{4}_{3}vcomp_parameter_maps_{1}_cnv.fits'.format(self.region, self.rootPara,
                                                                                          self.paraDir, 1, self.line_root)
                self.paraCubes = itf.cubefit_wTauTexCnvRef(self.OneOneFile, singCompRef, downsampfactor=2, **kwargs)

            else:
                self.paraCubes = itf.cubefit(self.OneOneFile, downsampfactor=2, **kwargs)


        else:
            # Note: it may be better not to use the single component fit as our Guesses; less errors to propagate
            if False:
                #if os.path.exists(self.SingVParaFile) and self.linename == "oneone":
                guesses = mvf.make_guesses(self.SingVParaFile, n_comp = n_comp)
            else:
                guesses = None

            # perform regular fitting
            kwargs = {'ncomp':n_comp, 'paraname':self.NewParaFile, 'modname':self.ModelFile, 'chisqname':self.ChisqFile,
                      'guesses':guesses, 'errmap11name':self.RMSFile, 'multicore':multicore, 'snr_min':snr_min,
                      'mask_function':mask_function, 'linename':self.linename}
            self.paraCubes = mvf.cubefit_gen(cube11name = self.OneOneFile, **kwargs)

        '''
        mvf.get_multiV_models(paraname=self.NewParaFile, refcubename=self.OneOneFile, n_comp = n_comp,
                                  savename = self.SepVModelFile, snrname = self.SNRFile, rms = 0.15, rmspath=self.RMSFile)
        '''
        self.end_timing()


    def calc_aic(self, n_comp1=1, n_comp2=2):
        # calculate the aic values over the same spectral windows for two given models

        modNameSp = self.ModelFile.split("comp_model")
        modpath1 = '{0}{1}{2}{3}'.format(modNameSp[0][:-1], n_comp1, "comp_model", modNameSp[1])
        modpath2 = '{0}{1}{2}{3}'.format(modNameSp[0][:-1], n_comp2, "comp_model", modNameSp[1])
        aiccpath = "{0}/{1}_NH3_{4}_{2}v{3}comp_aicc.fits".format(self.cleanParaDir, self.region, n_comp2, n_comp1,
                                                                  self.line_root)
        lnkpath = "{0}/{1}_NH3_{4}_{2}v{3}comp_aicc_lnK.fits".format(self.cleanParaDir, self.region, n_comp2, n_comp1,
                                                                  self.line_root)

        aic.fits_comp_AICc(self.OneOneFile, modpath1, modpath2, aiccpath, lnkpath)


    def calc_chisq(self, n_comp1=1, n_comp2=2):
        # calculate the reduced chi-squared values over the same spectral windows two given models
        modNameSp = self.ModelFile.split("comp_model")
        modpath1 = '{0}{1}{2}{3}'.format(modNameSp[0][:-1], n_comp1, "comp_model", modNameSp[1])
        modpath2 = '{0}{1}{2}{3}'.format(modNameSp[0][:-1], n_comp2, "comp_model", modNameSp[1])
        chisqpath = "{0}/{1}_NH3_{4}_{2}v{3}comp_redchisq.fits".format(self.cleanParaDir, self.region, n_comp2, n_comp1,
                                                                       self.line_root)

        aic.fits_comp_chisq(self.OneOneFile, modpath1, modpath2, chisqpath, reduced = True)


    def fit_2comp(self, fit1comp=True, multicore=8, snr_min=5.0, iterfit=True):
        # note: this method doesn't seem to provide much of an improvement over the current method of guessing
        kwargs = {'multicore':multicore, 'snr_min':snr_min, 'mask_function':None, 'iterfit':iterfit}
        if fit1comp:
            self.fit_cube(n_comp=1, **kwargs)
        self.fit_cube(n_comp=2, refOneCompCnv=True, **kwargs)
        self.calc_aic()
        self.calc_chisq()


    def clean_paramap(self):
        return None



#=======================================================================================================================
# here are some quick utilities functions

def make_dir(dirpath):
    # create directories if they don't exist
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


#=======================================================================================================================

'''
def run_L1688(n_comp=2):
    # for testing purpose
    region='L1688'
    multicore=8
    linename = "oneone"
    snr_min=5.0
    reg = Region(region, root='DR1_rebase3', linename=linename)
    reg.OneOneFile = '{2}/{0}/DR1/{0}_NH3_{3}_{1}_trim.fits'.format(reg.region, reg.root, reg.cubeDir, reg.line_root)
    reg.RMSFile = '{2}/{0}/DR1/{0}_NH3_{3}_{1}_rms_QA_trim.fits'.format(reg.region, reg.root, reg.cubeDir, reg.line_root)
    reg.SingVParaFile = None
    reg.fit_cube(n_comp=n_comp, multicore=multicore, snr_min=snr_min, mask_function = None, iterfit=True)
    return reg
'''

def run(region='L1448', multicore=8, root='all_rebase3'):
    # wrapper to run individual regions on a go
    regOb = Region(region, root=root)
    regOb.fit_cube(n_comp=2, multicore=multicore, snr_min=5.0, mask_function = None, iterfit=True)
    regOb.fit_cube(n_comp=1, multicore=multicore, snr_min=5.0, mask_function= None, iterfit=True)
    regOb.calc_aic()
    regOb.calc_chisq()
    return regOb

def DR1_run(region='NGC1333', multicore=8, linename = "oneone", snr_min=5.0):

    # start timing
    start_time = time.time()

    # for running on DR1 regions
    reg = Region(region, root='DR1_rebase3', linename=linename)

    reg.OneOneFile = '{2}/{0}/DR1/{0}_NH3_{3}_{1}_trim.fits'.format(reg.region, reg.root, reg.cubeDir, reg.line_root)
    reg.RMSFile = '{2}/{0}/DR1/{0}_NH3_{3}_{1}_rms_QA_trim.fits'.format(reg.region, reg.root, reg.cubeDir, reg.line_root)
    reg.SingVParaFile = None

    reg.fit_cube(n_comp=1, multicore=multicore, snr_min=snr_min, mask_function = None, iterfit=True)
    reg.fit_cube(n_comp=2, multicore=multicore, snr_min=snr_min, mask_function = None, iterfit=True)
    reg.calc_aic()
    reg.calc_chisq()

    #reg.fit_2comp(fit1comp=False, multicore=multicore, snr_min=snr_min, iterfit=True)

    elapsed_time = time.time() - start_time
    # print elapsed_time
    print "total time ellapsed"
    print time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    return reg


def super_run(linename = "oneone"):
    # note: likely hard on memory, especially gc.collect is ont used

    #DR1_run(region='L1688', multicore=8, linename=linename)
    DR1_run(region='NGC1333', multicore=14, linename=linename)

    #special_run(region='L1448', linename="oneone")
    #special_run(region='L1448', linename="twotwo")
    #special_run(region='OrionB_NGC2023-2024')
    '''
    run(region='OrionA')
    run(region='IC5146')
    run(region='OrionB_NGC2068-2071')
    run(region='Serpens_Aquila')
    run(region='L1688')
    '''

def special_run(region='L1448', multicore=8, linename = "oneone"):
    # for rebase that does not have other first look properties
    regOb = Region(region, root='base_all_rebase3', linename=linename)
    # use Jared's multi-rebased file

    regOb.OneOneFile = '{2}/{0}/{0}_NH3_{3}_{1}.fits'.format(regOb.region, 'all_rebase_multi', regOb.cubeDir, regOb.line_root)

    regOb.fit_cube(n_comp=2, multicore=multicore, snr_min=3.0, mask_function = None, iterfit=True)
    regOb.fit_cube(n_comp=1, multicore=multicore, snr_min=3.0, mask_function = None, iterfit=True)
    regOb.calc_aic()
    regOb.calc_chisq()
    return regOb





'''
# currenlty I plan to do the cleaning locally on mcychen
def clean():
    clean_map.clean(oriname, newname, snrname = snrname, one_v_map = one_v_name, aic_maps = aicname,
          fill_plane1 = False, mergesim = False, rm_sml_obj = True, sort_method = "None")
'''