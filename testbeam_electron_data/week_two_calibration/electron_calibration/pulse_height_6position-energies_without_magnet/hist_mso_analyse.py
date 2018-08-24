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
from math import *
from scipy import integrate
#import inspect, os
import scipy as sp
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import sys

# INPUT
script_name = sys.argv[0]
data_name = str(sys.argv[1])

print "Starting script:", script_name
print "Arguments:", data_name

# DATA
bins, counts = np.loadtxt(data_name, usecols=(0, 1), delimiter=',').T

# remove first and last bin
bins = bins[1:-1]
counts = counts[1:-1]

# fill bins, since there are empty
#first_bin = 40. # nVs
#last_bin = -360 # nVs
#bins = np.linspace(first_bin, last_bin, len(bins), endpoint=False)

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
#print data_info

##########
# PLOT
fig, ax = plt.subplots(figsize=(4, 3))#, dpi=100)
fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)

# plot histogram
plt.bar(bins, counts, width=binning,
        color='b', lw=0)
props = dict(boxstyle='square', facecolor='white')
plt.text(0.05, 0.95, data_info, fontsize=8, transform = ax.transAxes,
            verticalalignment='top', horizontalalignment='left', bbox=props)

# plot fit
x_fit = xdata
y_fit = fitfunc(x_fit, *para)
plt.plot(x_fit, y_fit, ls='-', lw=2, alpha=0.8, color='red')

# options
plt.yscale('log')
plt.ylim(1., 1e4)
plt.xlabel(r'pulse height [V]')
plt.ylabel(r'counts [#]')
plt.xlim(bins[0], bins[-1])

# Show plot, save results
save_name = script_name[:-3] + '_' + data_name[:-4] + ".pdf"
plt.savefig(save_name)
print "evince", save_name, "&"


# save parameter
# np.savez('summarize' + data_in, rn=rn, sn_3sigma=sn_3sigma, mu=mu, d_mu=d_mu, si=si, d_si=d_si, sigma3=sigma3, e_res=e_res, d_e_res=d_e_res, signal_counts_3sigma=signal_counts_3sigma, signal_counts_fit=signal_counts_fit, fit_start=fit_start, fit_end=fit_end, a=a, d_a=d_a, b=b, d_b=d_b, noise_counts_3sigma=noise_counts_3sigma, noise_fit_start=noise_fit_start, noise_fit_end=noise_fit_end, data_in=data_in, voltage_binning=voltage_binning, counts_binning=counts_binning, total_counts=total_counts)
