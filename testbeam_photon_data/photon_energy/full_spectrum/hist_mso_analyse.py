'''plot and analyse histograms from MSO scope

Usage:
    hist_mso_analyse.py (--input=<input>)

Options:
    --input=<input>             scope data
    -h --help                   show usage of this script
    -v --version                show the version of this script
'''

import numpy as np
import yaml
from docopt import docopt

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

import bremsstrahlung_spectrum

# INPUT
fit = False
script_name = sys.argv[0]

arguments = docopt(__doc__, version='test')
data_name = arguments['--input']
print data_name

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

# fit / mean
if fit:
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
data_info = ('full photon spectrum' +'\n' +
        r'total counts = %.1f'%(total_counts))
#print data_info

##########
# PLOT
fig, ax = plt.subplots(figsize=(4, 3))#, dpi=100)
fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)

# plot histogram
plt.bar(bins, counts, width=binning,
        color='#FFA500', lw=0)
props = dict(boxstyle='square', facecolor='white')
plt.text(0.05, 0.95, data_info, fontsize=8, transform = ax.transAxes,
            verticalalignment='top', horizontalalignment='left', bbox=props)

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
