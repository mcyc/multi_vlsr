import numpy as np
from astropy.io import fits
from collections import defaultdict
from astropy.table import Table
from multiprocessing import Pool, cpu_count
import warnings

#import nh3_testcubes as ntc
import fast_iter_fit as fifit
import fit_2comp as f2p
#reload(ntc)
reload(fifit)
reload(f2p)

import sys, os, time
# add the parent directory to the paths
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import iterative_fit as itf

#-----------------------------------------------------------------------------------------------------------------------
# wrappers to run on different machines


def run_gb(nCubes=10000, nBorder=1, make_cubes=False):
    workDir = "/lustre/pipeline/scratch/GAS/images/MChen_FakeCubes"
    #cubeSubDir = "random_cubes"
    cubeSubDir = "random_cubes_wRadTran"
    outDir = '{}/{}'.format(workDir, cubeSubDir)

    if make_cubes:
        genDir = "/users/mchen/GitHub_Repos/LineFitting"
        sys.path.insert(0, genDir)
        import multiproc_wrapper as mw

        kwargs = {'nCubes': nCubes, 'nBorder':nBorder, 'noise_rms':0.1, 'output_dir':outDir, 'random_seed':42,
                  'linenames':['oneone'], 'n_cpu':None}

        mw.generate_cubes(**kwargs)

    tableName = "cube_test_results_wRadTran.txt"

    return run_tests(nCubes, workDir, cubeSubDir=cubeSubDir, tablename=tableName)


def run_on_mc(nCubes=100, nBorder=1, make_cubes=False):

    workDir = '/Users/mcychen/Desktop'
    cubeSubDir = "random_cubes_wRadTran"
    outDir = '{}/{}'.format(workDir, cubeSubDir)

    if make_cubes:
        #generate_cubes(nBorder, nCubes, workDir)
        genDir = "/Users/mcychen/Documents/GitRepos/LineFitting"
        sys.path.insert(0, genDir)
        import multiproc_wrapper as mw

        kwargs = {'nCubes': nCubes, 'nBorder':nBorder, 'noise_rms':0.1, 'output_dir':outDir, 'random_seed':42,
                  'linenames':['oneone'], 'n_cpu':None}

        mw.generate_cubes(**kwargs)

    tableName = "cube_test_results_wRadTran.txt"

    return run_tests(nCubes, workDir, cubeSubDir=cubeSubDir, tablename=tableName)

#-----------------------------------------------------------------------------------------------------------------------
# core functions

def run_tests(nCubes, workDir, cubeSubDir=None, tablename=None):
    # ignore warnings
    warnings.filterwarnings('ignore')

    if cubeSubDir is None:
        cubeDir = "{}/random_cubes".format(workDir)
    else:
        cubeDir = "{}/{}".format(workDir, cubeSubDir)

    if tablename is None:
        tableName = "{}/cube_test_results.txt".format(workDir)
    else:
        tableName = "{}/{}".format(workDir, tablename)

    dict_truepara = read_cubes(cubeDir=cubeDir, nCubes=nCubes)

    start_time = time.time()
    print("")
    print("############################################")
    print("------------- start fitting ----------------")
    results = run_fit(cubeDir=cubeDir, nCubes=nCubes)
    elapsed_time = time.time() - start_time
    print("-------- total runtime for cube fit --------")
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    dict_fitpara = sort_fit_results(results)
    dict_final = merge_two_dicts(dict_truepara, dict_fitpara)

    return write_table(dict_final, outname=tableName)


def read_cubes(cubeDir, nCubes):
    #
    truekwds = ['NCOMP', 'LOGN1', 'LOGN2', 'VLSR1', 'VLSR2', 'SIG1', 'SIG2', 'TKIN1', 'TKIN2', 'TMAX', 'RMS',
                'TMAX-1', 'TMAX-2', 'TAU1', 'TAU2']

    truepara = defaultdict(list)
    truepara['CUBE_ID'] = []

    nDigits = int(np.ceil(np.log10(nCubes)))

    for i in range(nCubes):
        cubename = cubeDir + '/random_cube_NH3_11_'+ '{0}'.format(i).zfill(nDigits) + '.fits'
        cube, hdr = fits.getdata(cubename, header=True)
        for key in truekwds:
            truepara[key].append(hdr[key])
        truepara['CUBE_ID'].append("{}".format(i).zfill(nDigits))

    return truepara


def write_table(dict, outname=None, **kwargs):
    #
    names = []
    data = []
    for key, value in dict.iteritems():
        names.append(key)
        data.append(value)

    table = Table(data, names=names)

    if outname is not None:
        table.write(outname, format='ascii', overwrite=True, **kwargs)

    return table


def run_fit(cubeDir, nCubes):
    # perform iternative fitting

    nDigits = int(np.ceil(np.log10(nCubes)))

    cubenames = []
    for i in range(nCubes):
        cubename = cubeDir + '/random_cube_NH3_11_'+ '{0}'.format(i).zfill(nDigits) + '.fits'
        cubenames.append(cubename)

    kwargs = {'paraname': None, 'snr_min':3, 'linename':"oneone"} #, 'multicore':1 'ncomp': n_comp,

    return f2p.run(cubenames, guesses_pp=None, kwargs_pp=kwargs, ncpu=None)
    # para1, err1, para2, err2, likelyhood, rms = f2p.run(cubenames, guesses_pp=None, kwargs_pp=kwargs, ncpu=None)
    # return para1, err1, para2, err2, likelyhood, rms


def sort_fit_results(results):
    '''
    #para1, err1, para2, err2, likelyhood

    fitkwds = ['NCOMP_FIT', 'VLSR1_FIT', 'VLSR2_FIT', 'SIG1_FIT', 'SIG2_FIT', 'TEX1_FIT', 'TEX2_FIT', 'TAU1_FIT',
               'TAU2_FIT', 'LN_K_21']
    '''

    kwds_paraCompA = ['VLSR1_FIT', 'SIG1_FIT', 'TEX1_FIT', 'TAU1_FIT']
    kwds_paraCompB = ['VLSR2_FIT', 'SIG2_FIT', 'TEX2_FIT', 'TAU2_FIT']
    kwds_paraCompAB = kwds_paraCompA + kwds_paraCompB

    kwds_errCompA = ['eVLSR1_FIT', 'eSIG1_FIT', 'eTEX1_FIT', 'eTAU1_FIT']
    kwds_errCompB = ['eVLSR2_FIT', 'eSIG2_FIT', 'eTEX2_FIT', 'eTAU2_FIT']
    kwds_errCompAB = kwds_errCompA + kwds_errCompB

    fitpara = defaultdict(list)

    for i, lkhood in enumerate(results[4]):

        fitpara['LN_K_21'].append(lkhood)
        fitpara['RMS_FIT'].append(results[5][i])

        if lkhood > 5:
            fitpara['NCOMP_FIT'].append(2)
            for j, key in enumerate(kwds_paraCompAB):
                fitpara[key].append(results[2][i][j])
            for j, key in enumerate(kwds_errCompAB):
                fitpara[key].append(results[3][i][j])

        else:
            fitpara['NCOMP_FIT'].append(1)
            for j, (keyA, keyB) in enumerate(zip(kwds_paraCompA, kwds_paraCompB)):
                fitpara[keyA].append(results[0][i][j])
                fitpara[keyB].append(np.nan)
            for j, (keyA, keyB) in enumerate(zip(kwds_errCompA, kwds_errCompB)):
                fitpara[keyA].append(results[1][i][j])
                fitpara[keyB].append(np.nan)
    return fitpara


def merge_two_dicts(x, y):
    # merges two dictionaries into one
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z





