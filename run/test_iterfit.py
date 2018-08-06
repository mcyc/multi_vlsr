__author__ = 'mcychen'

import sys, os
# add the parent directory to the paths
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import multi_v_fit as mvf
reload(mvf)
import iterative_fit as itf
reload(itf)
import test_multiv as testmv

#=======================================================================================================================

def do():
    l1 = "oneone"
    kwarg = {'version':'lowC1_xlowC2', 'SNR1':'low', 'SNR2':'xlow'}
    run(l1, **kwarg)


def run(linename="oneone", version = "medC1_lowC2", SNR1="med", SNR2="low"):

    baseDir = "/Users/mcychen/Documents/Data/GAS_NH3"

    paraDir = "{0}/mock_paraMaps/{1}".format(baseDir, version)
    if not os.path.exists(paraDir):
        os.makedirs(paraDir)

    cubeDir = "{0}/mock_rebase/{1}".format(baseDir,  version)
    if not os.path.exists(cubeDir):
        os.makedirs(cubeDir)

    if linename == "oneone":
        line_root = "11"
    elif linename == "twotwo":
        line_root = "22"
    else:
        line_root = linename

    cubename = "{0}/mock_NH3_{1}_2vcomp_{2}_cube.fits".format(cubeDir, line_root, version)
    realparaname = "{0}/mock_NH3_{1}_2vcomp_{2}_trueparameter_maps.fits".format(paraDir, line_root, version)
    peakname = "{0}/mock_NH3_{1}_2vcomp_{2}_peaktemps.fits".format(cubeDir, line_root, version)

    if False:
        # make a fake cube for testing
        if linename == "oneone":
            tex1, tau1 = itf.mock_textau_11(SNR=SNR1)
            tex2, tau2 = itf.mock_textau_11(SNR=SNR2)
        elif linename == "twotwo":
            tex1, tau1 = itf.mock_textau_22(SNR=SNR1)
            tex2, tau2 = itf.mock_textau_22(SNR=SNR2)
        # generate a fake cube
        kwarg = {'tex1':tex1, 'tau1':tau1, 'tex2':tex2, 'tau2':tau2}
        testmv.fake_cube(fname = cubename, paraname = realparaname, linename = linename, **kwarg)

    paraname = "{0}/mock_NH3_{1}_2vcomp_{2}_parameter_maps.fits".format(paraDir, line_root, version)
    modname = "{0}/mock_NH3_{1}_2vcomp_{2}_modelcube.fits".format(cubeDir, line_root, version)

    if True:

        #dir = "/Users/mcychen/Documents/Data/GAS_NH3/mock_rebase/lowC1_xlowC2"
        #cubename = "{0}/mock_NH3_11_2vcomp_lowC1_xlowC2_cube.fits".format(dir)

        #paraDir = "/Users/mcychen/Documents/Data/GAS_NH3/mock_paraMaps/lowC1_xlowC2/"
        #paraname = "{0}/mock_NH3_11_2vcomp_lowC1_xlowC2_parameter_maps_refined.fits".format(paraDir)

        # supply the fitted parameter to the convovled cube
        if True:
            # if the convolved paramter already excited,
            conv_paraname = "{0}_refined.fits".format(os.path.splitext(paraname)[0], "parameter_maps")
            kwargs = {'ncomp':2, 'paraname':paraname, 'modname':None, 'chisqname':None, 'guesses':None, 'errmap11name':None,
                  'multicore':3, 'mask_function':None, 'snr_min':3.0, 'linename':"oneone", 'conv_paraname':conv_paraname}

        else:
            kwargs = {'ncomp':2, 'paraname':paraname, 'modname':None, 'chisqname':None, 'guesses':None, 'errmap11name':None,
                  'multicore':3, 'mask_function':None, 'snr_min':3.0, 'linename':"oneone"}


        # fit the fake cube with 2 velocity component models, iteratively
        pcube = itf.cubefit(cubename, downsampfactor=2, **kwargs)

    if False:
        figDir = "{0}/figures".format(paraDir)
        if not os.path.exists(figDir):
            os.makedirs(figDir)
        testmv.plot_vel_fit_accuracy(realparaname, paraname, saveFigDir=figDir, saveFigRoot="NH3_{0}".format(line_root))

    return None
