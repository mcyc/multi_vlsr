__author__ = 'mcychen'

#=======================================================================================================================
import os
import numpy as np

import UltraCube as UCube

#=======================================================================================================================


class Region(object):

    def __init__(self, cubeDir, cubeName, paraNameRoot, paraDir=None, cnv_cubepath=None, cnv_factor=2):

        self.cubeDir = cubeDir
        self.cubePath = "{}/{}".format(self.cubeDir, cubeName)

        self.paraNameRoot = paraNameRoot

        if paraDir is None:
            self.paraDir = "{}/para_maps".format(self.cubeDir)
        else:
            self.paraDir = paraDir

        if not os.path.exists(self.paraDir):
            os.makedirs(self.paraDir)

        self.ucube = UCube.UltraCube(self.cubePath)

        # for convolving cube
        self.cnv_factor = cnv_factor

        if cnv_cubepath is None:
            root = "conv{0}Xbeam".format(int(np.rint(self.cnv_factor)))
            self.cnv_cubePath = "{0}_{1}.fits".format(os.path.splitext(self.cubePath)[0], root)
        else:
            self.cnv_cubePath = cnv_cubepath

        self.cnv_para_paths ={}


    def get_convolved_cube(self, update=True, **kwargs):

        if update or (not os.path.isfile(self.cnv_cubePath)):
            convolve_cube(self, **kwargs)

        self.ucube_cnv = UCube.UltraCube(self.cnv_cubePath)


    def get_convolved_fits(self, ncomp, update=False, **kwargs):

        for nc in ncomp:
            if not str(nc) in self.cnv_para_paths:
                self.cnv_para_paths[str(nc)] = '{}_{}vcomp_conv2Xbeam.fits'.format(reg.paraNameRoot, nc)

        if update:
            fit_cnv_cube(reg, ncomp, snr_min=3, **kwargs)

        for nc in ncomp:
            self.ucube_cnv.load_model_fit(self.cnv_para_paths[str(nc)], nc)





def convolve_cube(reg, cnv_cubepath=None, cnv_factor=None, edgetrim_width=None):

    if not cnv_cubepath is None:
        reg.cnv_cubePath = cnv_cubepath

    if not cnv_factor is None:
        reg.cnv_factor = cnv_factor

    reg.ucube.convolve_cube(factor=reg.cnv_factor, savename=reg.cnv_cubePath, edgetrim_width=edgetrim_width)



def fit_cnv_cube(reg, ncomp, snr_min=3, **kwargs):
    #if not hasattr(reg, 'ucube_cnv'):

    for nc in ncomp:
        reg.ucube_cnv.fit_cube(ncomp=[nc], snr_min=snr_min, **kwargs)
        if not str(nc) in reg.cnv_para_paths:
            reg.cnv_para_paths[str(nc)] = '{}_{}vcomp_conv2Xbeam.fits'.format(reg.paraNameRoot, nc)
        UCube.save_model_fit(reg.ucube_cnv.pcubes[str(nc)], reg.cnv_para_paths[str(nc)], nc)


def fit_cnv_cube(reg, ncomp, snr_min=3, **kwargs):
    #if not hasattr(reg, 'ucube_cnv'):

    for nc in ncomp:
        reg.ucube_cnv.fit_cube(ncomp=[nc], snr_min=snr_min, **kwargs)
        if not str(nc) in reg.cnv_para_paths:
            reg.cnv_para_paths[str(nc)] = '{}_{}vcomp_conv2Xbeam.fits'.format(reg.paraNameRoot, nc)
        UCube.save_model_fit(reg.ucube_cnv.pcubes[str(nc)], reg.cnv_para_paths[str(nc)], nc)





def test():
    dirName = "/Users/mcychen/Desktop/UCube_Test/"
    cubepath = "{}/orionA_test_cube.fits".format(dirName)
    fit_2comp(cubepath, ncomp=2, edgetrim_width=5)



def fit_2comp(cubepath, downsampfactor=2, edgetrim_width=5, cnv_cubepath=None):

    # read the cube file
    ucube = UCube.UltraCube(cubepath)



    # fit 1 & 2 comp for the convolved cubes
    ucube_cnv = UCube.UltraCube(cnv_cubepath)
    ucube_cnv.fit_cube(ncomp=[1, 2], snr_min=3)
    UCube.save_model_fit(orion_cnv.pcubes['1'], '{}/orionA_test_para_1vcomp_conv2Xbeam.fits'.format(paraDir), 1)
    UCube.save_model_fit(orion_cnv.pcubes['2'], '{}/orionA_test_para_2vcomp_conv2Xbeam.fits'.format(paraDir), 2)
