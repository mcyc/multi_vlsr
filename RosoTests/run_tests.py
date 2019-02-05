import numpy as np
from astropy.io import fits
from collections import defaultdict
from astropy.table import Table
from multiprocessing import Pool, cpu_count


import nh3_testcubes as ntc
import fast_iter_fit as fifit
reload(ntc)
reload(fifit)

import sys, os #, errno, time
# add the parent directory to the paths
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import iterative_fit as itf
#-----------------------------------------------------------------------------------------------------------------------




def read_cubes(cubeDir, nCubes):

    truekwds = ['NCOMP', 'LOGN1', 'LOGN2', 'VLSR1', 'VLSR2', 'SIG1', 'SIG2', 'TKIN1', 'TKIN2', 'TMAX', 'RMS']
    truepara = defaultdict(list)

    nDigits = int(np.ceil(np.log10(nCubes)))

    for i in range(nCubes):
        cubename = cubeDir + '/random_cube_NH3_11_'+ '{0}'.format(i).zfill(nDigits) + '.fits'
        cube, hdr = fits.getdata(cubename, header=True)
        for key in truekwds:
            truepara[key].append(hdr[key])

    return truepara


def write_table(dict, outname=None, **kwargs):

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
    import fit_2comp as f2p
    reload(f2p)
    # perform iternative fitting

    nDigits = int(np.ceil(np.log10(nCubes)))

    cubenames = []
    for i in range(nCubes):
        cubename = cubeDir + '/random_cube_NH3_11_'+ '{0}'.format(i).zfill(nDigits) + '.fits'
        cubenames.append(cubename)

    kwargs = {'paraname': None, 'snr_min':3, 'linename':"oneone"} #, 'multicore':1 'ncomp': n_comp,

    para1, err1, para2, err2, likelyhood = f2p.run(cubenames, guesses_pp=None, kwargs_pp=kwargs, ncpu=None)
    return para1, err1, para2, err2, likelyhood


def run(nBorder=1, nCubes=6):

    workDir = '/Users/mcychen/Desktop'
    cubeDir = "{}/random_cubes".format(workDir)
    tableName = "{}/cube_test_results.txt".format(workDir)

    if False:
        generate_cubes(nBorder, nCubes, cubeDir)


    #tab_truepara = read_cubes(cubeDir=outDir, nCubes=nCubes)

    #run_fit(cubeDir=outDir, nCubes=nCubes, n_comp=1)

    #return write_table(tab_truepara, outname=tableName)

    return run_fit(cubeDir=cubeDir, nCubes=nCubes)


def generate_cubes(nBorder, nCubes, cubeDir):
    # generating nCubes number of test cubes
    kwargs = {'nCubes':nCubes, 'nBorder':nBorder, 'noise_rms':0.1, 'output_dir':cubeDir, 'random_seed':None,
              'TwoTwoLine':False}
    ntc.generate_cubes(**kwargs)


def test(nBorder=2, nCubes=4):
    workDir = '/Users/mcychen/Desktop'
    outDir = "{}/random_cubes".format(workDir)

    return run_fit(cubeDir=outDir, nCubes=nCubes, n_comp=1)

