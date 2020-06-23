__author__ = 'mcychen'

import sys, os
# add the parent directory to the paths
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import mad_std

import multi_v_fit as mvf
reload(mvf)
import iterative_fit as itf
reload(itf)
import test_multiv as testmv

#=======================================================================================================================

def do():
    l1 = "oneone"
    #l2 = "twotwo"

    if False:
        kwargs = {'version': 'lowC1_xlowC2', 'SNR1': 'low', 'SNR2': 'xlow'}
        # do a test on wide-v-sep case study
        kwargs['v1_extraoff'] = 1.95
        kwargs['recover_wide'] = True
        kwargs['makeMockCube'] = True

        testmv.run(l1, **kwargs)

    if True:
        kwargs = {'version': 'lowC1_lowC2', 'SNR1': 'low', 'SNR2': 'low'}
        # do a test on wide-v-sep case study
        kwargs['v1_extraoff'] = 1.95
        kwargs['recover_wide'] = True
        kwargs['makeMockCube'] = True

        testmv.run(l1, **kwargs)


    if False:
        kwarg = {'version':'lowC1_xlowC2', 'SNR1':'low', 'SNR2':'xlow'}
        #testmv.run(l1, **kwarg)
        run(l1, **kwarg)

    if False:
        kwarg = {'version':'medC1_lowC2', 'SNR1':'med', 'SNR2':'low'}
        #testmv.run(l1, **kwarg)
        run(l1, **kwarg)

    if False:
        kwarg = {'version':'medC1_xlowC2', 'SNR1':'med', 'SNR2':'xlow'}
        #testmv.run(l1, **kwarg)
        run(l1, **kwarg)

    if False:
        kwarg = {'version':'lowC1_xxlowC2', 'SNR1':'low', 'SNR2':'xxlow'}
        #testmv.run(l1, **kwarg)
        run(l1, **kwarg)

        kwarg = {'version':'medC1_xxlowC2', 'SNR1':'med', 'SNR2':'xxlow'}
        #testmv.run(l1, **kwarg)
        run(l1, **kwarg)

        kwarg = {'version':'lowC1_lowC2', 'SNR1':'low', 'SNR2':'low'}
        #testmv.run(l1, **kwarg)
        run(l1, **kwarg)

        kwarg = {'version':'medC1_medC2', 'SNR1':'med', 'SNR2':'med'}
        testmv.run(l1, **kwarg)
        run(l1, **kwarg)

    if False:
        kwarg = {'version':'highC1_lowC2', 'SNR1':'high', 'SNR2':'low'}
        testmv.run(l1, **kwarg)
        run(l1, **kwarg)

    if False:
        kwarg = {'version':'highC1_medC2', 'SNR1':'high', 'SNR2':'med'}
        testmv.run(l1, **kwarg)
        run(l1, **kwarg)




def run(linename="oneone", version = "medC1_lowC2", SNR1="med", SNR2="low", makeMockCube=False, fit1comp=True):

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

    if makeMockCube:
        # make a fake cube for testing
        if linename == "oneone":
            tex1, tau1 = testmv.mock_textau_11(SNR=SNR1)
            tex2, tau2 = testmv.mock_textau_11(SNR=SNR2)
        elif linename == "twotwo":
            tex1, tau1 = testmv.mock_textau_22(SNR=SNR1)
            tex2, tau2 = testmv.mock_textau_22(SNR=SNR2)
        # generate a fake cube
        kwarg = {'tex1':tex1, 'tau1':tau1, 'tex2':tex2, 'tau2':tau2}
        testmv.fake_cube(fname = cubename, paraname = realparaname, linename = linename, **kwarg)


    ncomp=2
    paraname = "{0}/mock_NH3_{1}_{2}vcomp_{3}_parameter_maps.fits".format(paraDir, line_root, ncomp, version)
    #modname = "{0}/mock_NH3_{1}_{2}vcomp_{3}_modelcube.fits".format(cubeDir, line_root, ncomp, version)

    if fit1comp:
        # perform the 1 comp first

        ncomp=1
        paraname = "{0}/mock_NH3_{1}_{2}vcomp_{3}_parameter_maps.fits".format(paraDir, line_root, ncomp, version)

        kwargs = {'ncomp': ncomp, 'paraname': paraname, 'modname': None, 'chisqname': None, 'guesses': None,
              'errmap11name': None, 'multicore': 5, 'mask_function': None, 'snr_min': 3.0, 'linename': "oneone",
              'momedgetrim': False}
        pcube = itf.cubefit(cubename, downsampfactor=2, **kwargs)


        # now perform the 2 comp fit using the tau tex guess from the 1 comp fit
        singCompRef = "{0}_cnv.fits".format(os.path.splitext(paraname)[0])

        ncomp=2
        paraname = "{0}/mock_NH3_{1}_{2}vcomp_{3}_parameter_maps.fits".format(paraDir, line_root, ncomp, version)

        kwargs = {'ncomp': ncomp, 'paraname': paraname, 'modname': None, 'chisqname': None, 'guesses': None,
              'errmap11name': None, 'multicore': 5, 'mask_function': None, 'snr_min': 3.0, 'linename': "oneone",
              'momedgetrim': False}

        pcube = itf.cubefit_wTauTexCnvRef(cubename, singCompRef, downsampfactor=2, **kwargs)

    else:
    # run the fitting routine
        # supply the fitted parameter to the convovled cube

        if False:
            # if the convolved paramter already excited,
            conv_paraname = "{0}_cnv.fits".format(os.path.splitext(paraname)[0], "parameter_maps")
            kwargs = {'ncomp':2, 'paraname':paraname, 'modname':None, 'chisqname':None, 'guesses':None,
                      'errmap11name':None, 'multicore':5, 'mask_function':None, 'snr_min':3.0, 'linename':"oneone",
                      'conv_paraname':conv_paraname, 'momedgetrim':False}

        else:
            kwargs = {'ncomp':2, 'paraname':paraname, 'modname':None, 'chisqname':None, 'guesses':None,
                      'errmap11name':None, 'multicore':5, 'mask_function':None, 'snr_min':3.0, 'linename':"oneone", 'momedgetrim':False}


        # fit the fake cube with 2 velocity component models, iteratively
        pcube = itf.cubefit(cubename, downsampfactor=2, **kwargs)

    if True:
        print paraname
        print realparaname
        figDir = "{0}/figures".format(paraDir)

        if not os.path.exists(figDir):
            os.makedirs(figDir)
        #itername = "{0}_iter.fits".format(os.path.splitext(paraname)[0], "parameter_maps")
        #plot_vel_fit_accuracy(realparaname, paraname, itername, saveFigDir=figDir, saveFigRoot="NH3_{0}".format(line_root))
        plot_vel_fit_accuracy(realparaname, paraname, paraname, saveFigDir=figDir, saveFigRoot="NH3_{0}".format(line_root))


    return None


#=======================================================================================================================

def plot_vel_fit_accuracy(name_realp, name_fitp, name_itrfit, saveFigDir="",  saveFigRoot=""):

    para_rl, hdr_rl = fits.getdata(name_realp, header=True)
    para_ft, hdr_ft = fits.getdata(name_fitp, header=True)
    para_itr, hdr_itr = fits.getdata(name_itrfit, header=True)

    # remove the failed fits
    para_ft[para_ft == 0.0] = np.nan
    para_itr[para_itr == 0.0] = np.nan


    '''
    # if the first component model fits the real second component better, swap it with the second component
    # note: this may be an issue for cases where two components have a very similar velocity
    swap = np.abs(para_ft[0] - para_rl[2]) < np.abs(para_ft[4] - para_rl[2])
    # note: check to see if the following operation does "overide" parts of the information
    para_ft[:4][:,swap], para_ft[4:8][:,swap]  = para_ft[4:8][:,swap], para_ft[:4][:,swap]
    '''

    # plot the 'real' error of the second component fit

    if False:
        # the fit error vs "real" error
        plt.clf()
        plt.scatter(para_ft[12], np.abs(para_ft[4] - para_rl[2]), s=3)
        plt.xlabel("fits error")
        plt.ylabel("real error")

    if False:
        # accuracy of the vlsr fit vs. vlsr seperation from the bright component
        plt.clf()
        plt.scatter(para_rl[2], para_ft[4] - para_rl[2], s=3)
        plt.xlabel(r"$\Delta$v$_{lsr}$ between the two components (km s$^{-1}$)")
        plt.ylabel(r"Fit and actual v$_{lsr}$ difference (km s$^{-1}$)")
        plt.savefig("{0}/{1}_vlsrErr_vs_deltaV_scatter.pdf".format(saveFigDir, saveFigRoot))

    if False:
        # accuracy of the vlsr fit vs. vlsr seperation from the bright component
        plt.clf()
        plt.scatter(para_rl[3], para_ft[4] - para_rl[2], s=3)
        plt.xlabel(r"Second Component $\sigma_{v}$ (km s$^{-1}$)")
        plt.ylabel(r"Difference between fit and actual v$_{lsr}$ (km s$^{-1}$)")
        plt.savefig("{0}/{1}_vlsrErr_vs_deltaSigma_scatter.pdf".format(saveFigDir, saveFigRoot))

    if False:
        # plot the histogram of the fit errors
        plt.clf()
        plt.hist((para_ft[0] - para_rl[0]).ravel(), 50, range=(-0.4,0.4), normed=False, histtype = "stepfilled", color="0.75")
        plt.hist((para_ft[4] - para_rl[2]).ravel(), 50, range=(-0.4,0.4), normed=False, histtype = "step")
        plt.legend(["rear component","front component"], frameon=False)
        plt.ylabel("Number of pixels")
        plt.xlabel(r"Difference between fit and actual v$_{lsr}$ (km s$^{-1}$)")
        plt.savefig("{0}/{1}_vlsrErr_histo.pdf".format(saveFigDir, saveFigRoot))

    if False:
        # plot the histogram of the fit errors relative to the estimated errors

        plt.clf()
        diff1 = (para_ft[0] - para_rl[0])/para_ft[8]
        diff2 = (para_ft[4] - para_rl[2])/para_ft[12]
        plt.hist(diff1.ravel(), 50, range=(-5,5), normed=False, histtype = "step")
        #plt.hist(diff2.ravel(), 50, range=(-5,5), normed=False, histtype = "step")

        #plt.clf()
        diff_itr1 = (para_itr[0] - para_rl[0])/para_itr[8]
        diff_itr2 = (para_itr[4] - para_rl[2])/para_itr[12]
        plt.hist(diff_itr1.ravel(), 50, range=(-5,5), normed=False, histtype = "step")
        #plt.hist(diff_itr2.ravel(), 50, range=(-5,5), normed=False, histtype = "step")

        plt.legend(["rear component","front component"], frameon=False)
        plt.title("Accuracy in the 1,1 fits")
        plt.ylabel("Number of pixels")
        plt.xlabel(r"Difference between fit and actual v$_{lsr}$ over the estimated error")
        plt.savefig("{0}/{1}_vlsrErrRelEst_histo_iter2.pdf".format(saveFigDir, saveFigRoot))


    if True:
        # histogram comparison between relative error of the regular fit vs. absolute fit
        # default matplotlib colors
        cBlue = "#1f77b4"
        cOrng = "#ff7f0e"
        hRg = (-0.2,0.2)
        numbin = 30

        diff1 = (para_ft[0] - para_rl[0]).ravel()
        diff2 = (para_ft[4] - para_rl[2]).ravel()
        diff_itr1 = (para_itr[0] - para_rl[0]).ravel()
        diff_itr2 = (para_itr[4] - para_rl[2]).ravel()

        # histogram comparison between the relative error of the regular fit vs. absolute fit
        plt.clf()
        plt.hist(diff1, numbin, range=hRg, histtype="stepfilled", alpha=0.5, color=cBlue)
        plt.hist(diff2, numbin, range=hRg, histtype="stepfilled", alpha=0.5, color=cOrng)
        plt.hist(diff_itr1, numbin, range=hRg, histtype="step",  color=cBlue)
        plt.hist(diff_itr2, numbin, range=hRg, histtype="step", color=cOrng)
        #diff_itr1 = (para_itr[0] - para_rl[0])/para_itr[8]
        #diff_itr2 = (para_itr[4] - para_rl[2])/para_itr[12]

        rdigit = 3
        lgnd1 = "reg. rear; $\sigma_{mad} = $" + str(np.round(mad_std(diff1[np.isfinite(diff1)]),rdigit))
        lgnd2 = "reg. front; $\sigma_{mad} = $" + str(np.round(mad_std(diff2[np.isfinite(diff2)]),rdigit))
        lgnd3 = "iter. rear; $\sigma_{mad} = $" + str(np.round(mad_std(diff_itr1[np.isfinite(diff_itr1)]),rdigit))
        lgnd4 = "iter. front; $\sigma_{mad} = $" + str(np.round(mad_std(diff_itr2[np.isfinite(diff_itr2)]),rdigit))

        #median1 = np.round(np.median(diff_itr1[np.isfinite(diff_itr1)]/mad_std(diff1[np.isfinite(diff1)])),4)
        #median2 = np.round(np.median(diff_itr2[np.isfinite(diff_itr2)]/mad_std(diff2[np.isfinite(diff2)])),4)
        median1 = np.round(np.median(diff_itr1[np.isfinite(diff_itr1)]),4)
        median2 = np.round(np.median(diff_itr2[np.isfinite(diff_itr2)]),4)
        plt.title("Accuracy in the 1,1 fits. Iter. median: {0} & {1}".format(median1, median2))

        plt.legend([lgnd1, lgnd2, lgnd3, lgnd4], frameon=False)
        plt.ylabel("Number of pixels")
        plt.xlabel(r"v$_{lsr}$ fit accuracy (km s$^{-1}$)")
        plt.savefig("{0}/{1}_vlsrErr_histo_iter2.pdf".format(saveFigDir, saveFigRoot))

    if True:
        # KDE comparison between the relative error of the regular fit vs. absolute fit

        # relative accuracy from regular fitting
        diff1 = (para_ft[0] - para_rl[0])/para_ft[8]
        diff2 = (para_ft[4] - para_rl[2])/para_ft[12]
        diff1 = diff1[np.isfinite(diff1)].ravel()
        diff2 = diff2[np.isfinite(diff2)].ravel()

        # relative accuracy from the iterative fitting
        diff_itr1 = (para_itr[0] - para_rl[0])/para_itr[8]
        diff_itr2 = (para_itr[4] - para_rl[2])/para_itr[12]

        diff_itr1 = diff_itr1[np.isfinite(diff_itr1)].ravel()
        diff_itr2 = diff_itr2[np.isfinite(diff_itr2)].ravel()

        numbin = 50
        uplim = 5
        lowlim = -5
        bins = np.linspace(lowlim, uplim, numbin)

        plt.clf()
        plot_kde(diff1, bins)
        plot_kde(diff_itr1, bins)
        legend_1 = "reg. fit; $\sigma_{MAD} = $" + str(np.round(mad_std(diff1), 2))
        legend_2 = "iter. fit; $\sigma_{MAD} = $" + str(np.round(mad_std(diff_itr1), 2))
        plt.legend([legend_1,legend_2], frameon=False)
        plt.axvspan(-1, 1, alpha=0.5, color='0.75')
        plt.axvline(x=0, linestyle='--', color="0.75", alpha=0.75)
        plt.title("Accuracy in the 1,1 fits relative to the estimated errors")
        plt.ylabel("Fraction of pixels")
        plt.xlabel(r"Difference between fit and actual v$_{lsr}$ over the estimated error")
        plt.savefig("{0}/{1}_vlsrErrRelEst_kde_iter_rearComp.pdf".format(saveFigDir, saveFigRoot))

        plt.clf()
        plot_kde(diff2, bins)
        plot_kde(diff_itr2, bins)
        legend_1 = "reg. fit; $\sigma_{MAD} = $" + str(np.round(mad_std(diff2), 2))
        legend_2 = "iter. fit; $\sigma_{MAD} = $" + str(np.round(mad_std(diff_itr2), 2))
        plt.legend([legend_1, legend_2], frameon=False)
        plt.axvline(x=0, linestyle='--', color="0.75", alpha=0.75)
        #plt.axvline(x=1.1775, linestyle='--', color="0.75")
        #plt.axvline(x=-1.1775, linestyle='--', color="0.75")
        plt.axvspan(-1, 1, alpha=0.5, color='0.75')
        plt.title("Accuracy in the 1,1 fits relative to the estimated errors")
        plt.ylabel("Fraction of pixels")
        plt.xlabel(r"Difference between fit and actual v$_{lsr}$ over the estimated error")
        plt.savefig("{0}/{1}_vlsrErrRelEst_kde_iter_frontComp.pdf".format(saveFigDir, saveFigRoot))




def plot_kde(x, bins, **kwargs):
    # a quick way to plot KDE in a similar fashion to a histogram
    from sklearn.neighbors import KernelDensity
    binwidth = bins[1] - bins[0]
    X_plot = np.linspace(bins.min(), bins.max(), 1000)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=binwidth*2).fit(np.reshape(x,(x.shape[0],1)))
    log_dens = kde.score_samples(X_plot)
    plt.plot(X_plot[:, 0], np.exp(log_dens), **kwargs)
