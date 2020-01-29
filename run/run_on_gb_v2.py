__author__ = 'mcychen'

'''
This is a wrapper to run on the Green Bank server for the multi_vlsr package
'''
import sys, os, errno, time
import time

import master_fitter as mf

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

        self.root = root
        if rootPara is None:
            self.rootPara = self.root

        # defining paths to imaged data
        self.OneOneFile = '{2}/{0}/{0}_NH3_{3}_{1}.fits'.format(self.region, self.root, self.cubeDir, self.line_root)
        self.RMSFile = '{2}/{0}/{0}_NH3_{3}_{1}_rms.fits'.format(self.region, self.root, self.cubeDir, self.line_root)
        self.SingVParaFile = '{2}/{0}/{0}_parameter_maps_{1}.fits'.format(self.region, self.rootPara, self.cubeDir)



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


def verify_package():
    # start timing
    start_time = time.time()

    # for running on DR1 regions
    reg = Region(region, root='DR1_rebase3', linename=linename)
    reg.OneOneFile = '{2}/{0}/DR1/{0}_NH3_{3}_{1}_trim.fits'.format(reg.region, reg.root, reg.cubeDir, reg.line_root)

    paraNameRoot = '{0}_NH3_{1}_{2}_para'.format(reg.region, reg.line_root, reg.rootPara)
    uReg = mf.Region(cubePath=reg.OneOneFile, paraNameRoot=paraNameRoot, paraDir=reg.paraDir)
    uReg.master_2comp_fit(snr_min=3)

    elapsed_time = time.time() - start_time
    # print elapsed_time
    print "total time ellapsed"
    print time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    return reg, uReg

