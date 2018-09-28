#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

### for spectrum
def spectrum_slice(kmin, kmax, kinit, thickness, radiation_length):
    return thickness/radiation_length * (
            4./3. *np.log(kmax/kmin) -
            4*(kmax-kmin)/(3.*kinit) +
            (kmax**2 - kmin**2)/(2. * kinit**2))

def gauss(xvalues, position, width, height):
    return height * np.exp(-0.5*(xvalues - position)**2/width**2)

def spectrum(momentum_initial, thickness, X0,
        momentum_start, momentum_end, bins, res_a, res_b,
        height):
    momenta_limits = np.linspace(momentum_start, momentum_end, bins)
    # shift bins in the center & remove the last bin 
    binning = momenta_limits[1] - momenta_limits[0]
    momenta = momenta_limits[:-1] + binning/2
    spectra_height = spectrum_slice(momenta_limits[:-1], momenta_limits[1:],
            kinit=5., thickness=thickness, radiation_length=X0)
    # resolution
    linear = lambda x, *p: p[0] * x + p[1]
    parameter = [res_a, res_b]
    momenta_res = linear(momenta, *parameter)
    # spectrum
    spectra = np.zeros(len(momenta))
    for index, value in enumerate(momenta[momenta <= momentum_initial]):
        #print value, spectra_height_cu[index]
        spectra += gauss(momenta, value,
                width=momenta_res, height=spectra_height[index])
    spectra = height * spectra
    return spectra


############################################3
if __name__ == "__main__":
    ###
    script_name = sys.argv[0]

    ### x-values
    momenta_limits = np.linspace(0.1, 6., 100)
    print len(momenta_limits)
    # shift bins in the center & remove the last bin 
    binning = momenta_limits[1] - momenta_limits[0]
    momenta = momenta_limits[:-1] + binning/2
    #print len(momenta)

    # implement width according to the measurement: 1GeV+-13.6% to 3.6+-5.6%
    energy       = np.array([1., 1.6, 2.0, 2.6, 3.0, 3.6])
    res_relative = np.array([0.1378, 0.0998, 0.0898, 0.0710, 0.0645, 0.0572])
    # fit
    fitfunc = lambda x, *p: p[0] * x + p[1]
    para0 = [0.05, 0.0]
    xdata = energy
    ydata = energy*res_relative
    dydata = np.sqrt(ydata); dydata = np.where(dydata > 0.0, dydata, 1) #; print dydata 
    para, cov = curve_fit(fitfunc, xdata, ydata, p0=para0, sigma=dydata)
    chi2 = np.sum(((ydata - fitfunc(xdata, *para)) / dydata)**2)
    chi2red = chi2 / (len(ydata)-len(para))
    print "para", para
    xdata_fit = momenta
    ydata_fit = fitfunc(xdata_fit, *para)
    momenta_res = ydata_fit
    # plot
    fig, ax = plt.subplots(figsize=(4, 3))#, dpi=100)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
    plt.plot(energy, energy*res_relative, 'xk')
    plt.plot(xdata_fit, ydata_fit, '--k')
    plt.xlabel(r'energy [GeV]')
    plt.ylabel(r'resolution [GeV]')
    save_name = script_name[:-3] + '_energy_resolution' + ".pdf"
    plt.savefig(save_name)
    print "evince", save_name, "&"

    ###
    cu_thickness=0.3
    cu_xo=1.436

    air_thickness=165.
    air_xo=30390.


    #########
    # Copper Converter
    spectra_cu = spectrum(momentum_initial=5., thickness=cu_thickness, X0=cu_xo,
            momentum_start=0.1, momentum_end=6.0, bins=100, res_a=para[0], res_b=para[1],
            height=1.0)
    print len(spectra_cu)


    #########
    # Air Converter
    spectra_air = spectrum(momentum_initial=5., thickness=air_thickness, X0=air_xo,
            momentum_start=0.1, momentum_end=6.0, bins=100, res_a=para[0], res_b=para[1],
            height=1.0)
    print len(spectra_air)



    ##########
    # PLOT
    fig, ax = plt.subplots(figsize=(4, 3))#, dpi=100)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)

    plt.bar(momenta, spectra_cu, width=binning,
            color='#FFA500', lw=0)

    plt.bar(momenta, spectra_air, width=binning,
            color='#808080', lw=0)

    # options
    plt.yscale('log')
    #plt.xlim(bins[0], bins[-1])
    plt.ylim(1e-4, 1.)
    plt.xlabel(r'energy [GeV]')
    plt.ylabel(r'counts')

    # Show plot, save results
    #save_name = script_name[:-3] + ".png"
    #plt.savefig(save_name)
    save_name = script_name[:-3] + ".pdf"
    plt.savefig(save_name)
    print "evince", save_name, "&"
    #plt.show()
