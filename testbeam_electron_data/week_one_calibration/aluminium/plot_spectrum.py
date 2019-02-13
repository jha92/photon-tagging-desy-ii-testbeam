#! /usr/bin/env python
'''plot and analyse histograms from MSO scope

Usage:
    plot_spectrum.py
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# INPUT
name = 'Aluminium'
color_name = 'firebrick'
script_name = sys.argv[0]
data_name = str(sys.argv[1])

print "Starting script:", script_name
print "Arguments:", data_name

# DATA
with open(data_name,'r') as infile:
    data = infile.readlines()[1:-1]

bins  = np.array([float(row.split(",")[0]+"."+row.split(",")[1]) for row in data])
counts = np.array([float(row.split(",")[2]) for row in data])

# make bins positive 
bins = bins * -1.
# shift bins in the center
binning = bins[1] - bins[0]
bins = bins - binning/2
# normalize counts
counts = counts / 500

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
dydata = np.sqrt(ydata); dydata = np.where(dydata > 0.0, dydata, 1)
para, cov = curve_fit(fitfunc, xdata, ydata, p0=para0, sigma=dydata)
# chi**2
chi2 = np.sum(((ydata - fitfunc(xdata, *para)) / dydata)**2)
chi2red = chi2 / (len(ydata)-len(para))

total_counts = np.sum(counts)


##########
props = dict(boxstyle='square', facecolor='white')
bin_low = 0
bin_up = 7

### plot distribution
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
plt.bar(bins, counts, width=binning, color=color_name)
ax.legend([name], title=('%1.1f GeV'%float(data_name[:1])), fontsize='small')
# options
plt.yscale('log')
plt.ylim(1., 1e4)
plt.xlabel(r'pulse height [V]')
plt.ylabel(r'counts [#]')
plt.xlim(bin_low, bin_up)
# save
save_name = script_name[:-3] + '_' + data_name[:-4] + ".png"
plt.savefig(save_name)


### plot distribution with fit
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
bar = plt.bar(bins, counts, width=binning, color=color_name)
ax.legend([name], title=('%1.1f GeV'%float(data_name[:1])), fontsize='small')
fit_info = (r'total counts = %1d'%(total_counts) + '\n' +
            r'mean fit = %.2f'%(para[0]) + '\n' +
            r'sigma fit = %.2f'%(abs(para[1]))
	    )
plt.text(0.6, 0.78, fit_info, fontsize=8, transform = ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
x_fit = xdata
y_fit = fitfunc(x_fit, *para)
fit = plt.plot(x_fit, y_fit, ls='--', lw=1, alpha=0.8, color='red')
# options
plt.yscale('log')
plt.ylim(1., 1e4)
plt.xlabel(r'pulse height [V]')
plt.ylabel(r'counts [#]')
plt.xlim(bin_low, bin_up)
# save
save_name = script_name[:-3] + '_' + data_name[:-4] + "_fit.png"
plt.savefig(save_name)
# save parameter
with open('fitparameter.txt','a') as outfile:
	outfile.write('%.1f,%.4f,%.4f\n'%(float(data_name[:1]),para[0],abs(para[1])))


### plot normalized distribution
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
counts_norm = counts/total_counts
plt.bar(bins, counts_norm, width=binning, color=color_name)
ax.legend([name], title=('%1.1f GeV'%float(data_name[:1])), fontsize='small')
plt.yscale('log')
plt.ylim(1e-5, 3e-1)
plt.xlabel(r'pulse height [V]')
plt.ylabel(r'normalized counts')
plt.xlim(bin_low, bin_up)
save_name = script_name[:-3] + '_' + data_name[:-4] + "_norm.png"
plt.savefig(save_name)
