import numpy as np
from astropy.io import fits
from collections import defaultdict
from astropy.table import Table

import nh3_testcubes as ntc
reload(ntc)
#-----------------------------------------------------------------------------------------------------------------------

def run(nBorder=2, nCubes=2):

    workDir = '/Users/mcychen/Desktop'
    outDir = "{}/random_cubes".format(workDir)

    kwargs = {'nCubes':nCubes, 'nBorder':nBorder, 'noise_rms':0.1, 'output_dir':outDir, 'random_seed':None,
              'TwoTwoLine':False}

    #ntc.generate_cubes(**kwargs)
    truepara = read_cubes(cubeDir=outDir,nCubes=nCubes)
    return write_table(truepara)


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


def write_table(dict):

    names = []
    data = []
    for key, value in dict.iteritems():
        names.append(key)
        data.append(value)

    t = Table(data, names=names)

    return t


