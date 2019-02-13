#! /usr/bin/env python
'''plot calibration of scope data

Usage:
    plot_calibration.py
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# INPUT
name = 'Air'
color_name = 'deepskyblue'

# DATA
energy,volt,sigma = np.loadtxt('fitparameter.txt', delimiter=',', usecols=(0,1,2), unpack=True)

# FIT linear
fitfunc = lambda x, *p: p[0] + p[1] * x
xdata = volt
ydata = energy
dydata = np.sqrt(ydata); dydata = np.where(dydata > 0.0, dydata, 1)
para, cov = curve_fit(fitfunc, xdata, ydata, p0=[0.,1.], sigma=dydata)


##########
props = dict(boxstyle='square', facecolor='white')

### plot calibration curve
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
plt.errorbar(volt, energy, yerr=sigma, marker='o', markersize=2, linestyle='None', color=color_name)
data_info = ('Calibration with ' + name)
plt.text(0.05, 0.95, data_info, fontsize=8, transform = ax.transAxes,
            verticalalignment='top', horizontalalignment='left', bbox=props)
# options
plt.ylim(0., 4.5)
plt.xlabel(r'pulse height [V]')
plt.ylabel(r'energy [GeV]')
plt.xlim(0.,4.)
# save
plt.savefig("calibration.png")


### plot calibration curve with fit
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
plt.errorbar(volt, energy, yerr=sigma, marker='o', markersize=2, linestyle='None', color=color_name)
data_info = ('Calibration with ' + name)
plt.text(0.05, 0.95, data_info, fontsize=8, transform = ax.transAxes,
            verticalalignment='top', horizontalalignment='left', bbox=props)
fit_info = (r'E_GeV = %.4f + pulse_V * %.4f'%(para[0],para[1]))
plt.text(0.3, 0.1, fit_info, fontsize=8, transform = ax.transAxes,
            verticalalignment='top', horizontalalignment='left')
x_fit = volt
y_fit = fitfunc(x_fit, *para)
plt.plot(x_fit, y_fit, ls='-', lw=1, alpha=0.8, color='red')
# options
plt.ylim(0., 4.5)
plt.xlabel(r'pulse height [V]')
plt.ylabel(r'electron energy [GeV]')
plt.xlim(0.,4.)
# save
plt.savefig("calibration_fit.png")
