
import numpy as np

import matplotlib.pyplot as plt

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
x, y, e = np.loadtxt(data_name, usecols=(0, 1, 2), delimiter=' ').T

# remove first bin
x = x
y = y / 10
e = e / 10

# array information 
#print x, counts, len(x)

##########
# PLOT
fig, ax = plt.subplots(figsize=(10, 6))#, dpi=100)
fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)

# plot
plt.errorbar(x, y, yerr = e, fmt='o', markersize=5, capsize=5)

# options
#plt.yscale('log')

plt.title('Lead-Glass Detector Bias Voltage')
plt.ylim(3600, 5400)
plt.xlabel(r'Negative Bias Voltage [V]')
plt.ylabel(r'Count Rate [$s^-1$]')
plt.xlim(x[0]-50, x[-1]+50)
plt.grid(linestyle='--')

plt.xticks(np.arange(1600, 2150, 50))

# Show plot, save results
save_name =  "bias_voltage.pdf"
plt.savefig(save_name)
plt.show()
print "evince", save_name, "&"


# save parameter
# np.savez('summarize' + data_in, rn=rn, sn_3sigma=sn_3sigma, mu=mu, d_mu=d_mu, si=si, d_si=d_si, sigma3=sigma3, e_res=e_res, d_e_res=d_e_res, signal_counts_3sigma=signal_counts_3sigma, signal_counts_fit=signal_counts_fit, fit_start=fit_start, fit_end=fit_end, a=a, d_a=d_a, b=b, d_b=d_b, noise_counts_3sigma=noise_counts_3sigma, noise_fit_start=noise_fit_start, noise_fit_end=noise_fit_end, data_in=data_in, voltage_binning=voltage_binning, counts_binning=counts_binning, total_counts=total_counts)
