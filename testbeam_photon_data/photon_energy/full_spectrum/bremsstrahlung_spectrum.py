#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
script_name = sys.argv[0]

def spectrum(kmin, kmax, energy, thickness, radiation_length):
    return thickness/radiation_length * (
            4./3. *np.log(kmax/kmin) -
            4*(kmax-kmin)/(3.*energy) +
            (kmax**2 - kmin**2)/(2. * energy**2))

momenta = np.linspace(0., 6., 100)
spectra = spectrum(momenta[:-1], momenta[1:], 6., 0.3, 1.436)

print len(momenta), len(spectra)

# shift bins in the center
binning = momenta[1] - momenta[0]
momenta = momenta + binning/2
# remove the last bin 
momenta = momenta[:-1]
print len(momenta)

##########
# PLOT
fig, ax = plt.subplots(figsize=(4, 3))#, dpi=100)
fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)

plt.bar(momenta, spectra, width=binning,
        color='#FFA500', lw=0)

# options
plt.yscale('log')
#plt.ylim(1., 1e4)
plt.xlabel(r'energy [GeV]')
plt.ylabel(r'counts')
#plt.xlim(bins[0], bins[-1])

# Show plot, save results
save_name = script_name[:-3] + ".png"
save_name = script_name[:-3] + ".pdf"
plt.savefig(save_name)
print "evince", save_name, "&"

exit()


