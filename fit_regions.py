__author__ = 'mcychen'

import multi_v_fit as mvf
import time
import os, errno


#=======================================================================================================================


def run_on_gb(region='L1448', multicore=8):
    # fourier has 8 cores
    start_time = time.time()
    print("fitting 1 component")
    cubefit_gb(region=region, snr_min=5.0, multicore=multicore, n_comp = 1)
    print("--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    print("fitting 2 component")
    cubefit_gb(region=region, snr_min=5.0, multicore=multicore, n_comp = 2)
    print("--- %s seconds ---" % (time.time() - start_time))





def run(reg):
    # copiapo and lascar should be able to support up to 12 cores
    #return cubefit_mc(region='NGC1333', multicore = 3)


    # NOTE: I'll recommand snr_min=5.0 instead... which is actually sufficient in recovering a lot of places with the
    # modeled peak SNR >3
    if reg == 'NGC1333':
        return cubefit_uvic(region='NGC1333', snr_min=3.0, multicore = 8)

    if reg == 'OrionA':
        return cubefit_orion(region='OrionA', snr_min=3.0, multicore = 10)


def cubefit_orion(region='OrionA', snr_min=5.0, multicore=1, mask_function = None):
    # runnint the cubefit on the UVic server
    paraDir = '/nfs/lican13/home/mcychen/Documents/GAS_Project/data/DR1_paraMaps'
    cubeDir = '/nfs/lican13/home/mcychen/Documents/GAS_Project/data/DR1_rebase3'
    root = 'DR1_rebase3_trim'
    rootPara = 'DR1_rebase3_flag'

    OneOneFile = '{2}/{0}/{0}_NH3_11_{1}.fits'.format(region,root,cubeDir)
    RMSFile = '{2}/{0}/{0}_NH3_11_DR1_rebase3_rms_QA_trim.fits'.format(region,root,cubeDir)
    SingVParaFile = '{2}/{0}/{0}_parameter_maps_{1}.fits'.format(region, rootPara, paraDir)
    NewParaFile = '{2}/{0}/{0}_2vcomp_parameter_maps_{1}.fits'.format(region, rootPara, paraDir)

    ModelFile = '{2}/{0}/{0}_NH3_11_{1}_model.fits'.format(region,root,cubeDir)
    ChisqFile = '{2}/{0}/{0}_2vcomp_chisq_{1}.fits'.format(region, rootPara, paraDir)

    SNRFile = '{2}/{0}/{0}_2vcomp_SNR_eachV_{1}.fits'.format(region, rootPara, paraDir)
    SepVModelFile = '{2}/{0}/{0}_NH3_11_{1}_sepVModel.fits'.format(region,root,cubeDir)

    guesses = mvf.make_guesses(SingVParaFile)

    pcubes = mvf.cubefit(cube11name = OneOneFile, paraname = NewParaFile, modname = ModelFile, chisqname = ChisqFile,
                       guesses = guesses, errmap11name = RMSFile, multicore = multicore, snr_min=snr_min,
                       mask_function = mask_function)

    mvf.get_multiV_models(paraname=NewParaFile, refcubename=OneOneFile, n_comp = 2, savename = SepVModelFile,
                                 snrname = SNRFile, rms = 0.15)

    return pcubes



def run_on_tupungato():
    # run cube fit on Tupungato (has 12 cores and 18 CPUs???)
    # cubefit_uvic(region='NGC1333', snr_min=5.0, multicore=12, n_comp = 2)

    start_time = time.time()
    cubefit_uvic(region='NGC1333', snr_min=5.0, multicore=12, n_comp = 1)
    print("--- %s seconds ---" % (time.time() - start_time))



def cubefit_gb(region='NGC1333', snr_min=5.0, multicore=1, mask_function = None, n_comp = 2):
    # running the cubefit on green bank server

    # defining directories
    rootDir = "/lustre/pipeline/scratch/GAS"
    cubeDir = "{0}/images".format(rootDir)
    paraDir = "{0}/{1}/paraMaps_MChen".format(cubeDir, region)
    modelDir = "{0}/{1}/multi_v_models_MChen".format(cubeDir, region)
    try:
        os.makedirs(paraDir)
        os.makedirs(modelDir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    root = 'base_all_rebase3'
    rootPara = 'base_all_rebase3'

    # defining paths to imaged data
    OneOneFile = '{2}/{0}/{0}_NH3_11_{1}.fits'.format(region,root,cubeDir)
    RMSFile = '{2}/{0}/{0}_NH3_11_{1}_rms.fits'.format(region,root,cubeDir)
    SingVParaFile = '{2}/{0}/{0}_parameter_maps_{1}.fits'.format(region, rootPara, cubeDir)

    # defining paths to the imaged data
    NewParaFile = '{2}/{0}_{3}vcomp_parameter_maps_{1}.fits'.format(region, rootPara, paraDir, n_comp)
    ModelFile = '{2}/{0}_NH3_11_{1}_{3}comp_model.fits'.format(region, root, modelDir, n_comp)
    ChisqFile = '{2}/{0}_{3}vcomp_chisq_{1}.fits'.format(region, rootPara, paraDir, n_comp)

    SNRFile = '{2}/{0}_{3}vcomp_SNR_eachV_{1}.fits'.format(region, rootPara, paraDir, n_comp)
    SepVModelFile = '{2}/{0}_NH3_11_{1}_{3}VModel.fits'.format(region, root, modelDir, n_comp)


    if os.path.exists(SingVParaFile):
        guesses = mvf.make_guesses(SingVParaFile, n_comp = n_comp)
    else:
        guesses = None

    # fit the data
    pcubes = mvf.cubefit_gen(cube11name = OneOneFile, ncomp = n_comp , paraname = NewParaFile, modname = ModelFile,
                             chisqname = ChisqFile, guesses = guesses, errmap11name = RMSFile, multicore = multicore,
                             snr_min=snr_min, mask_function = mask_function)

    # determine the signal to noise ratio
    # mchen: a way to determine the average rms based on the RMSFile for individual regions will be better
    mvf.get_multiV_models(paraname=NewParaFile, refcubename=OneOneFile, n_comp = n_comp, savename = SepVModelFile,
                                     snrname = SNRFile, rms = 0.15, rmspath=RMSFile)

    return pcubes



def cubefit_uvic(region='NGC1333', snr_min=5.0, multicore=1, mask_function = None, n_comp = 2):
    # runnint the cubefit on the UVic server
    paraDir = '/astro/mcychen/Documents/GAS_Project/data/DR1_paraMaps'
    cubeDir = '/astro/mcychen/Documents/GAS_Project/data/DR1_rebase3'
    root = 'DR1_rebase3'
    rootPara = 'DR1_rebase3'

    OneOneFile = '{2}/{0}/{0}_NH3_11_{1}.fits'.format(region,root,cubeDir)
    RMSFile = '{2}/{0}/{0}_NH3_11_{1}_rms.fits'.format(region,root,cubeDir)
    SingVParaFile = '{2}/{0}/{0}_parameter_maps_{1}_flag.fits'.format(region, rootPara, paraDir)
    NewParaFile = '{2}/{0}/{0}_{3}vcomp_parameter_maps_{1}.fits'.format(region, rootPara, paraDir, n_comp)

    ModelFile = '{2}/{0}/{0}_NH3_11_{1}_{3}comp_model.fits'.format(region,root,cubeDir, n_comp)
    ChisqFile = '{2}/{0}/{0}_{3}vcomp_chisq_{1}.fits'.format(region, rootPara, paraDir, n_comp)

    SNRFile = '{2}/{0}/{0}_{3}vcomp_SNR_eachV_{1}.fits'.format(region, rootPara, paraDir, n_comp)
    SepVModelFile = '{2}/{0}/{0}_NH3_11_{1}_{3}VModel.fits'.format(region,root,cubeDir, n_comp)


    guesses = mvf.make_guesses(SingVParaFile, n_comp = n_comp)

    pcubes = mvf.cubefit_gen(cube11name = OneOneFile, ncomp = n_comp , paraname = NewParaFile, modname = ModelFile, chisqname = ChisqFile,
                         guesses = guesses, errmap11name = RMSFile, multicore = multicore, snr_min=snr_min,
                         mask_function = mask_function)

    mvf.get_multiV_models(paraname=NewParaFile, refcubename=OneOneFile, n_comp = n_comp, savename = SepVModelFile,
                                     snrname = SNRFile, rms = 0.15)


    return pcubes






def cubefit_mc(region='NGC1333', snr_min=5.0, multicore=1, mask_function = None):
    # note, this needed to be updated to accept n_comp as one of the parameters (Oct 1, 2017)
    # runnint the cubefit on my own laptop

    paraDir = '/Users/mcychen/Documents/Data/GAS_NH3/DR1_paraMaps'
    cubeDir = '/Users/mcychen/Documents/Data/GAS_NH3/DR1_rebase3'
    root = 'DR1_rebase3'
    rootPara = 'DR1_rebase3'

    OneOneFile = '{2}/{0}/{0}_NH3_11_{1}.fits'.format(region,root,cubeDir)
    RMSFile = '{2}/{0}/{0}_NH3_11_{1}_rms.fits'.format(region,root,cubeDir)
    SingVParaFile = '{2}/{0}/{0}_parameter_maps_{1}_flag.fits'.format(region, rootPara, paraDir)
    NewParaFile = '{2}/{0}/{0}_2vcomp_parameter_maps_{1}.fits'.format(region, rootPara, paraDir)

    ModelFile = '{2}/{0}/{0}_NH3_11_{1}_model.fits'.format(region,root,cubeDir)
    ChisqFile = '{2}/{0}/{0}_2vcomp_chisq_{1}.fits'.format(region, rootPara, paraDir)

    SNRFile = '{2}/{0}/{0}_2vcomp_SNR_eachV_{1}.fits'.format(region, rootPara, paraDir)
    SepVModelFile = '{2}/{0}/{0}_NH3_11_{1}_sepVModel.fits'.format(region,root,cubeDir)

    '''
    guesses = mvf.make_guesses(SingVParaFile)

    pcubes = mvf.cubefit(cube11name = OneOneFile, paraname = NewParaFile, modname = ModelFile, chisqname = ChisqFile,
                       guesses = guesses, errmap11name = RMSFile, multicore = multicore, snr_min=snr_min,
                       mask_function = mask_function)
    '''

    return mvf.get_multiV_models(paraname=NewParaFile, refcubename=OneOneFile, n_comp = 2, savename = SepVModelFile,
                                 snrname = SNRFile, rms = 0.15)

