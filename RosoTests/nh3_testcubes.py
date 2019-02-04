import pyspeckit.spectrum.models.ammonia as ammonia
import pyspeckit.spectrum.models.ammonia_constants as nh3con
from pyspeckit.spectrum.units import SpectroscopicAxis as spaxis
import os
import sys
import numpy as np
import astropy.units as u
from astropy.io import fits
from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar
from astropy import log
log.setLevel('ERROR')

def generate_cubes(nCubes=2, nBorder=1, noise_rms=0.1, output_dir='random_cubes', random_seed=None, TwoTwoLine=False):
    """
    This places nCubes random cubes into the specified output directory
    Note: the TwoTwoLine condition can definitely be implemented more effeciently
    """

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    xarr11 = spaxis((np.linspace(-500, 499, 1000) * 5.72e-6
                     + nh3con.freq_dict['oneone'] / 1e9),
                    unit='GHz',
                    refX=nh3con.freq_dict['oneone'] / 1e9,
                    velocity_convention='radio', refX_unit='GHz')

    if TwoTwoLine:
        xarr22 = spaxis((np.linspace(-500, 499, 1000) * 5.72e-6
                         + nh3con.freq_dict['twotwo'] / 1e9), unit='GHz',
                        refX=nh3con.freq_dict['twotwo'] / 1e9,
                        velocity_convention='radio', refX_unit='GHz')

    nDigits = int(np.ceil(np.log10(nCubes)))
    if random_seed:
        np.random.seed(random_seed)
    nComps = np.random.choice([1, 2], nCubes)

    Temp1 = 8 + np.random.rand(nCubes) * 17
    Temp2 = 8 + np.random.rand(nCubes) * 17

    Voff1 = np.random.rand(nCubes) * 5 - 2.5
    Voff2 = np.random.rand(nCubes) * 5 - 2.5

    logN1 = 13 + 2 * np.random.rand(nCubes)
    logN2 = 13 + 2 * np.random.rand(nCubes)

    Width1NT = 0.3 * np.exp(0.5 * np.random.randn(nCubes))
    Width2NT = 0.3 * np.exp(0.5 * np.random.randn(nCubes))

    Width1 = np.sqrt(Width1NT + 0.08**2)
    Width2 = np.sqrt(Width2NT + 0.08**2)

    scale = np.array([[0.2, 0.1, 0.5, 0.01]])
    gradX1 = np.random.randn(nCubes, 4) * scale
    gradY1 = np.random.randn(nCubes, 4) * scale
    gradX2 = np.random.randn(nCubes, 4) * scale
    gradY2 = np.random.randn(nCubes, 4) * scale

    params1 = [{'ntot':14,
                'width':1,
                'xoff_v':0.0}] * nCubes
    params2 = [{'ntot':14,
                'width':1,
                'xoff_v':0.0}] * nCubes

    hdrkwds = {'BUNIT': 'K',
               'INSTRUME': 'KFPA    ',
               'BMAJ': 0.008554169991270138,
               'BMIN': 0.008554169991270138,
               'TELESCOP': 'GBT',
               'WCSAXES': 3,
               'CRPIX1': 2,
               'CRPIX2': 2,
               'CRPIX3': 500,
               'CDELT1': -0.008554169991270138,
               'CDELT2': 0.008554169991270138,
               'CDELT3': 5720.0,
               'CUNIT1': 'deg',
               'CUNIT2': 'deg',
               'CUNIT3': 'Hz',
               'CTYPE1': 'RA---TAN',
               'CTYPE2': 'DEC--TAN',
               'CTYPE3': 'FREQ',
               'CRVAL1': 0.0,
               'CRVAL2': 0.0,
               'LONPOLE': 180.0,
               'LATPOLE': 0.0,
               'EQUINOX': 2000.0,
               'SPECSYS': 'LSRK',
               'RADESYS': 'FK5',
               'SSYSOBS': 'TOPOCENT'}
    truekwds = ['NCOMP', 'LOGN1', 'LOGN2', 'VLSR1', 'VLSR2',
                'SIG1', 'SIG2', 'TKIN1', 'TKIN2']

    for i in ProgressBar(range(nCubes)):
        xmat, ymat = np.indices((2 * nBorder + 1, 2 * nBorder + 1))
        cube11 = np.zeros((xarr11.shape[0], 2 * nBorder + 1, 2 * nBorder + 1))

        if TwoTwoLine:
            cube22 = np.zeros((xarr22.shape[0], 2 * nBorder + 1, 2 * nBorder + 1))

        for xx, yy in zip(xmat.flatten(), ymat.flatten()):
            T1 = Temp1[i] * (1 + gradX1[i, 0] * (xx - 1)
                             + gradY1[i, 0] * (yy - 1)) + 5
            T2 = Temp2[i] * (1 + gradX2[i, 0] * (xx - 1)
                             + gradY2[i, 0] * (yy - 1)) + 5
            if T1 < 2.74:
                T1 = 2.74
            if T2 < 2.74:
                T2 = 2.74
            W1 = np.abs(Width1[i] * (1 + gradX1[i, 1] * (xx - 1)
                                     + gradY1[i, 1] * (yy - 1)))
            W2 = np.abs(Width2[i] * (1 + gradX2[i, 1] * (xx - 1)
                                     + gradY2[i, 1] * (yy - 1)))
            V1 = Voff1[i] + (gradX1[i, 2] * (xx - 1) + gradY1[i, 2] * (yy - 1))
            V2 = Voff2[i] + (gradX2[i, 2] * (xx - 1) + gradY2[i, 2] * (yy - 1))
            N1 = logN1[i] * (1 + gradX1[i, 3] * (xx - 1)
                             + gradY1[i, 3] * (yy - 1))
            N2 = logN2[i] * (1 + gradX2[i, 3] * (xx - 1)
                             + gradY2[i, 3] * (yy - 1))

            if nComps[i] == 1:
                spec11 = ammonia.cold_ammonia(xarr11, T1, ntot=N1, width=W1, xoff_v=V1)

            if nComps[i] == 2:
                spec11 = (ammonia.cold_ammonia(xarr11, T1, ntot=N1, width=W1, xoff_v=V1)
                          + ammonia.cold_ammonia(xarr11, T2, ntot=N2, width=W2, xoff_v=V2))

            cube11[:, yy, xx] = spec11
            Tmax11 = np.max(cube11[:, nBorder, nBorder])
            cube11 += np.random.randn(*cube11.shape) * noise_rms

            if TwoTwoLine:
                if nComps[i] == 1:
                    spec22 = ammonia.cold_ammonia(xarr22, T1, ntot=N1, width=W1, xoff_v=V1)

                if nComps[i] == 2:
                    spec22 = (ammonia.cold_ammonia(xarr22, T1, ntot=N1, width=W1, xoff_v=V1)
                              + ammonia.cold_ammonia(xarr22, T2, ntot=N2, width=W2, xoff_v=V2))

                cube22[:, yy, xx] = spec22
                Tmax22 = np.max(cube22[:, nBorder, nBorder])
                cube22 += np.random.randn(*cube22.shape) * noise_rms

        hdu11 = fits.PrimaryHDU(cube11)
        for kk in hdrkwds:
            hdu11.header[kk] = hdrkwds[kk]
            for kk, vv in zip(truekwds, [nComps[i], logN1[i], logN2[i],
                                         Voff1[i], Voff2[i], Width1[i], Width2[i],
                                         Temp1[i], Temp2[i]]):
                hdu11.header[kk] = vv
        hdu11.header['TMAX'] = Tmax11
        hdu11.header['RMS'] = noise_rms
        hdu11.header['CRVAL3'] = 23694495500.0
        hdu11.header['RESTFRQ'] = 23694495500.0
        hdu11.writeto(output_dir + '/random_cube_NH3_11_'
                      + '{0}'.format(i).zfill(nDigits)
                      + '.fits',
                      overwrite=True)

        if TwoTwoLine:
            hdu22 = fits.PrimaryHDU(cube22)

            for kk in hdrkwds:
                hdu22.header[kk] = hdrkwds[kk]
                for kk, vv in zip(truekwds, [nComps[i], logN1[i], logN2[i],
                                             Voff1[i], Voff2[i], Width1[i], Width2[i],
                                             Temp1[i], Temp2[i]]):
                    hdu22.header[kk] = vv
            hdu22.header['TMAX'] = Tmax22
            hdu22.header['RMS'] = noise_rms
            hdu22.header['CRVAL3'] = 23722633600.0
            hdu22.header['RESTFRQ'] = 23722633600.0
            hdu22.writeto(output_dir + '/random_cube_NH3_22_'
                          + '{0}'.format(i).zfill(nDigits) + '.fits',
                          overwrite=True)

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) > 1:
        generate_cubes(nCubes=int(sys.argv[1]))
    else:
        generate_cubes()
