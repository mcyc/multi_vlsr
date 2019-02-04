import numpy as np
from astropy.io import fits
from collections import defaultdict
from astropy.table import Table


import nh3_testcubes as ntc
import fast_iter_fit as fifit
reload(ntc)
reload(fifit)

import sys, os #, errno, time
# add the parent directory to the paths
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import iterative_fit as itf
#-----------------------------------------------------------------------------------------------------------------------

def run(nBorder=2, nCubes=2):

    workDir = '/Users/mcychen/Desktop'
    outDir = "{}/random_cubes".format(workDir)
    tableName = "{}/cube_test_results.txt".format(workDir)

    kwargs = {'nCubes':nCubes, 'nBorder':nBorder, 'noise_rms':0.1, 'output_dir':outDir, 'random_seed':None,
              'TwoTwoLine':False}

    #ntc.generate_cubes(**kwargs)
    tab_truepara = read_cubes(cubeDir=outDir, nCubes=nCubes)

    run_fit(cubeDir=outDir, nCubes=nCubes, n_comp=1)

    return write_table(tab_truepara, outname=tableName)


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



def run_fit(cubeDir, nCubes, n_comp=1):
    # perform iternative fitting

    nDigits = int(np.ceil(np.log10(nCubes)))

    for i in range(nCubes):
        cubename = cubeDir + '/random_cube_NH3_11_'+ '{0}'.format(i).zfill(nDigits) + '.fits'
        paraname = cubeDir + '/paraMaps_NH3_11_'+ '{0}'.format(i).zfill(nDigits) + '{}comp.fits'.format(n_comp)
        kwargs = {'ncomp': n_comp, 'paraname': paraname,
                  'guesses':None, 'multicore':1, 'snr_min':3, 'linename':"oneone"}

        '''
        'modname': self.ModelFile, 'chisqname': self.ChisqFile, 'mask_function': mask_function
        '''

        paraCubes = fifit.cubefit(cubename, downsampfactor=2, **kwargs)
        return paraCubes


def test(nBorder=2, nCubes=2):
    workDir = '/Users/mcychen/Desktop'
    outDir = "{}/random_cubes".format(workDir)

    return run_fit(cubeDir=outDir, nCubes=nCubes, n_comp=1)

