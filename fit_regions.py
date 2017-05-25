__author__ = 'mcychen'

import multi_v_fit as mvf
reload(mvf)

#=======================================================================================================================

def run(reg):
    # copiapo and lascar should be able to support up to 12 cores
    #return cubefit_mc(region='NGC1333', multicore = 3)

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
    RMSFile = '{2}/{0}/{0}_NH3_11_{1}_rms_QA_trim.fits'.format(region,root,cubeDir)
    SingVParaFile = '{2}/{0}/{0}_parameter_maps_{1}_flag.fits'.format(region, rootPara, paraDir)
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




def cubefit_uvic(region='NGC1333', snr_min=5.0, multicore=1, mask_function = None):
    # runnint the cubefit on the UVic server
    paraDir = '/nfs/lican13/home/mcychen/Documents/GAS_Project/data/DR1_paraMaps'
    cubeDir = '/nfs/lican13/home/mcychen/Documents/GAS_Project/data/DR1_rebase3'
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

    guesses = mvf.make_guesses(SingVParaFile)

    pcubes = mvf.cubefit(cube11name = OneOneFile, paraname = NewParaFile, modname = ModelFile, chisqname = ChisqFile,
                       guesses = guesses, errmap11name = RMSFile, multicore = multicore, snr_min=snr_min,
                       mask_function = mask_function)

    mvf.get_multiV_models(paraname=NewParaFile, refcubename=OneOneFile, n_comp = 2, savename = SepVModelFile,
                                 snrname = SNRFile, rms = 0.15)

    return pcubes
















def cubefit_mc(region='NGC1333', snr_min=5.0, multicore=1, mask_function = None):
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

