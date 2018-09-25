#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

###
script_name = sys.argv[0]

###
def spectrum_slice(kmin, kmax, energy_initial, thickness, radiation_length):
    return thickness/radiation_length * (
            4./3. *np.log(kmax/kmin) -
            4*(kmax-kmin)/(3.*energy_initial) +
            (kmax**2 - kmin**2)/(2. * energy_initial**2))

def gauss(xvalues, position, width, height):
    return height * np.exp(-0.5*(xvalues - position)**2/width**2)


###
cu_thickness=0.3
cu_xo=1.436

air_thickness=165.
air_xo=30390. 


### x-values
momenta_limits = np.linspace(0.1, 6., 100)
print len(momenta_limits)
# shift bins in the center & remove the last bin 
binning = momenta_limits[1] - momenta_limits[0]
momenta = momenta_limits[:-1] + binning/2
#print len(momenta)

#########
# Copper Converter

# counts
spectra_height_cu = spectrum_slice(momenta_limits[:-1], momenta_limits[1:],
        energy_initial=5., thickness=cu_thickness, radiation_length=cu_xo)
# TODO: implement width according to the measurement: 1GeV+-13.6% to 3.6+-5.6%


spectra_cu = np.zeros(len(momenta))
for index, value in enumerate(momenta[momenta <= 5.]):
    print value, spectra_height_cu[index]
    spectra_cu += gauss(momenta, value, width=0.2, height=spectra_height_cu[index])

#print len(spectra_cu)


#########
# Air Converter
#

# counts
spectra_height_air = spectrum_slice(momenta_limits[:-1], momenta_limits[1:],
        energy_initial=5., thickness=air_thickness, radiation_length=air_xo)


spectra_air = np.zeros(len(momenta))
for index, value in enumerate(momenta[momenta <= 5.]):
    print value, spectra_height_air[index]
    spectra_air += gauss(momenta, value, width=0.2, height=spectra_height_air[index])

#print len(spectra_air)




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
plt.ylim(1e-10, 1.)
plt.xlabel(r'energy [GeV]')
plt.ylabel(r'counts')
#plt.xlim(bins[0], bins[-1])

# Show plot, save results
#save_name = script_name[:-3] + ".png"
#plt.savefig(save_name)
#save_name = script_name[:-3] + ".pdf"
#plt.savefig(save_name)
#print "evince", save_name, "&"
plt.show()

exit()
