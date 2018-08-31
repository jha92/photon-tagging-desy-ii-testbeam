'''plot and analyse histograms from MSO scope

Usage:
    hist_mso_analyse.py [--configuration=<configuration>]

Options:
    --configuration=<configuration> yaml file
    -h --help                   show usage of this script
    -v --version                show the version of this script
'''

import numpy as np
import yaml
from docopt import docopt

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


#from math import *
#from scipy import integrate
#import inspect, os
#import scipy as sp
#from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import sys

# INPUT
script_name = sys.argv[0]

print "Starting script:", script_name

data_names = ['1.0_GeV_electrons_180822_153820.csv',
        '1.6_GeV_electrons_180822_154504.csv',
        '2.0_GeV_electrons_180822_154852.csv',
        '2.6_GeV_electrons_180822_155142.csv',
        '3.0_GeV_electrons_180822_155627.csv',
        '3.6_GeV_electrons_180822_155939.csv']


##########
# PLOT
fig, ax, = plt.subplots(figsize=(10, 6))#, dpi=100)
fig.subplots_adjust(left=0.17, right=0.95, top=0.9, bottom=0.17)

ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.set_axisbelow(True)

def plot_calibration(data_name, data_color):
    # DATA
    bins, counts = np.loadtxt(data_name, usecols=(0, 1), delimiter=',').T

    # remove first and last bin
    bins = bins[1:-1]
    counts = counts[1:-1]

    # make bins positive 
    bins = bins * -1.
    # shift bins in the center
    binning = bins[1] - bins[0]
    bins = bins - binning/2
    # normalize counts
    one_count = 511
    counts = counts / 511

    # FIT / Mean
    fitfunc = lambda x, *p: p[2] * np.exp(-0.5*(x-p[0])**2/p[1]**2)
    maximum_bin_index = np.where(np.max(counts)==counts)[0][0]
    mu_guess = bins[maximum_bin_index]
    si_guess = 1.
    norm_guess = np.max(counts)
    para0 = [mu_guess, si_guess, norm_guess] # mu, sigma, norm

    # fit 
    index_around = 12
    start_index = maximum_bin_index-index_around
    end_index = maximum_bin_index+index_around
    xdata = bins[start_index:end_index]
    ydata = counts[start_index:end_index]
    dydata = np.sqrt(ydata); dydata = np.where(dydata > 0.0, dydata, 1) #; print dy 
    para, cov = curve_fit(fitfunc, xdata, ydata, p0=para0, sigma=dydata)
    # chi**2
    chi2 = np.sum(((ydata - fitfunc(xdata, *para)) / dydata)**2)
    chi2red = chi2 / (len(ydata)-len(para))

    # array information 
    #print bins, counts, len(bins)
    total_counts = np.sum(counts)
    data_info = (data_name[:17].replace("_", " ") +'\n' +
            r'total counts = %.1f'%(total_counts) + '\n' +
            r'mean fit = %.2f'%(para[0]) + '\n' +
            r'sigma fit = %.2f'%(abs(para[1])))
    data_label = (r'%s: $%.2f \pm %.2f\, (%.1f \,\rm{perc.}) \, \rm{V}$'%(data_name[:7].replace("_", " "), para[0], abs(para[1]), abs(para[1])/para[0] * 100 ))
    #print data_info

    norm_factor = para[2] 

    # plot istogram
    # normalize
    counts = counts / norm_factor
    ax.bar(bins, counts, width=binning,
            color=data_color, lw=0)
    #props = dict(boxstyle='square', facecolor='white')
    #plt.text(para[0], 10., data_label, fontsize=8, #transform = ax.transAxes,
    #            verticalalignment='top', horizontalalignment='left',
    #            rotation=45)#, bbox=props)
    # plot fit
    #x_fit = bins #xdata 
    #y_fit = fitfunc(x_fit, *para)
    # normalize
    #y_fit = y_fit / norm_factor
    #plt.plot(x_fit, y_fit, ls='-', lw=2, color='#696969', label=data_label)

color_palette = ['#76EEC6',
        '#00CED1',
        '#00B2EE',
        '#009ACD',
        '#1874CD',
        '#104E8B']



for index, value in enumerate(data_names):
    plot_calibration(value, color_palette[index])

# options

# bottom x axis
new_tick_locations = np.array([1.18, 1.87, 2.34, 3.07, 3.55, 4.28])
ax.set_xlabel(r'Pulse Height [V]', fontsize=16)
ax.set_xticks(new_tick_locations)
ax.set_xlim(0, 5.4)

# second x axis
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
new_tick_labels = np.array([1.0, 1.6, 2.0, 2.6, 3.0, 3.6])
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(new_tick_labels)
ax2.set_xlabel(r"Beam Energy [GeV]", fontsize = 16)
ax2.set_ylabel(r"Counts [a.u.]", fontsize = 16)


#plt.title('Lead counter calibration with electrons', fontsize=16)
#plt.legend(fontsize=4)
plt.yscale('log')
plt.ylim(7e-2, 1e1)
plt.ylabel(r'Counts [a.u.]', fontsize = 16)
#plt.ylim(0, 1.6)
#plt.xlabel(r'Electron Energy', fontsize=16)

# Show plot, save results
save_name = script_name[:-3] + '_' + 'all' + ".png"
plt.savefig(save_name)
print "evince", save_name, "&"
plt.show()

# save parameter
# np.savez('summarize' + data_in, rn=rn, sn_3sigma=sn_3sigma, mu=mu, d_mu=d_mu, si=si, d_si=d_si, sigma3=sigma3, e_res=e_res, d_e_res=d_e_res, signal_counts_3sigma=signal_counts_3sigma, signal_counts_fit=signal_counts_fit, fit_start=fit_start, fit_end=fit_end, a=a, d_a=d_a, b=b, d_b=d_b, noise_counts_3sigma=noise_counts_3sigma, noise_fit_start=noise_fit_start, noise_fit_end=noise_fit_end, data_in=data_in, voltage_binning=voltage_binning, counts_binning=counts_binning, total_counts=total_counts)
