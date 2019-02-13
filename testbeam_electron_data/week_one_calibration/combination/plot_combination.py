#! /usr/bin/env python
'''plot combination of spectra

Usage:
    plot_combination.py
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

# INPUT: air
name_air = 'Air'
color_air = 'deepskyblue'
files_air = ('../air/1_0Gev_180817_123649.csv', '../air/2_0Gev_180817_130817.csv', '../air/3_0Gev_180817_132836.csv', '../air/4_0Gev_180817_135607.csv')

# INPUT: aluminium
name_alu = 'Aluminium'
color_alu = 'firebrick'
files_alu = ('../aluminium/1_Gev_180817_155507.csv', '../aluminium/2_Gev_180817_152859.csv', '../aluminium/3_Gev_180817_151158.csv', '../aluminium/4_Gev_180817_143816.csv')

# INPUT: air
name_tungsten = 'Tungsten'
color_tungsten = 'grey'
files_tungsten = ('../tungsten/1_Gev_180817_174123.csv', '../tungsten/2_Gev_180817_172120.csv', '../tungsten/3_Gev_180817_170321.csv', '../tungsten/4_Gev_180817_163242.csv')

p0 = 0.0083
p1 = 1.2949
calibfunc = lambda x: p0 + p1 * x
energy = (1.0, 2.0, 3.0, 4.0)

for idE, E in enumerate(energy):

	with open(files_air[idE],'r') as infile:
	    data_air = infile.readlines()[1:-1]
	with open(files_alu[idE],'r') as infile:
	    data_alu = infile.readlines()[1:-1]
	with open(files_tungsten[idE],'r') as infile:
	    data_tungsten = infile.readlines()[1:-1]

	bins_air  = np.array([float(row.split(",")[0]+"."+row.split(",")[1]) for row in data_air])
	counts_air = np.array([float(row.split(",")[2]) for row in data_air])
	bins_alu  = np.array([float(row.split(",")[0]+"."+row.split(",")[1]) for row in data_alu])
	counts_alu = np.array([float(row.split(",")[2]) for row in data_alu])
	bins_tungsten  = np.array([float(row.split(",")[0]+"."+row.split(",")[1]) for row in data_tungsten])
	counts_tungsten = np.array([float(row.split(",")[2]) for row in data_tungsten])

	# make bins positive 
	bins_air = bins_air * -1.
	bins_alu = bins_alu * -1.
	bins_tungsten = bins_tungsten * -1.
	# shift bins in the center
	binning = bins_air[1] - bins_air[0]
	bins_air = bins_air - binning/2
	bins_alu = bins_alu - binning/2
	bins_tungsten = bins_tungsten - binning/2
	# apply calibration
	binsE_air = calibfunc(bins_air)
	binsE_alu = calibfunc(bins_alu)
	binsE_tungsten = calibfunc(bins_tungsten)
	binningE = binsE_air[1] - binsE_air[0]
	# normalize counts by one_count
	counts_air = counts_air / 500
	counts_alu = counts_alu / 500
	counts_tungsten = counts_tungsten / 500
	# normalize total counts
	counts_air = counts_air / np.sum(counts_air)
	counts_alu = counts_alu / np.sum(counts_alu)
	counts_tungsten = counts_tungsten / np.sum(counts_tungsten)


	### plot normalized distributions
	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	plt.bar(bins_air, counts_air, width=binning, color=color_air)
	ax.legend(['Air'], title=(str(E)+' GeV'), fontsize='small')
	plt.yscale('log')
	plt.ylim(1e-5, 3e-1)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'normalized counts')
	plt.xlim(0, 7)
	save_name = str(E).replace(".","_")+"GeV_Air.png"
	plt.savefig(save_name)
	plt.close()

	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	plt.bar(bins_alu, counts_alu, width=binning, color=color_alu)
	ax.legend(['Aluminium'], title=(str(E)+' GeV'), fontsize='small')
	plt.yscale('log')
	plt.ylim(1e-5, 3e-1)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'normalized counts')
	plt.xlim(0, 7)
	save_name = str(E).replace(".","_")+"GeV_Aluminium.png"
	plt.savefig(save_name)
	plt.close()

	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	plt.bar(bins_tungsten, counts_tungsten, width=binning, color=color_tungsten)
	ax.legend(['Tungsten'], title=(str(E)+' GeV'), fontsize='small')
	plt.yscale('log')
	plt.ylim(1e-5, 3e-1)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'normalized counts')
	plt.xlim(0, 7)
	save_name = str(E).replace(".","_")+"GeV_Tungsten.png"
	plt.savefig(save_name)
	plt.close()

	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	bar1 = plt.bar(bins_air, counts_air, width=binning, color=color_air, alpha=0.5)
	bar2 = plt.bar(bins_alu, counts_alu, width=binning, color=color_alu, alpha=0.5)
	bar3 = plt.bar(bins_tungsten, counts_tungsten, width=binning, color=color_tungsten, alpha=0.5)
	ax.legend( (bar1,bar2,bar3), ('Air','Aluminium','Tungsten'), title=(str(E)+' GeV'), fontsize='small')
	plt.yscale('log')
	plt.ylim(1e-5, 3e-1)
	plt.xlabel(r'pulse height [V]')
	plt.ylabel(r'normalized counts')
	plt.xlim(0, 7)
	save_name = str(E).replace(".","_")+"GeV_overlay.png"
	plt.savefig(save_name)
	plt.close()

	### plot normalized distributions with energy scale
	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	plt.bar(binsE_air, counts_air, width=binningE, color=color_air)
	ax.legend(['Air'], title=(str(E)+' GeV'), fontsize='small')
	plt.yscale('log')
	plt.ylim(1e-5, 3e-1)
	plt.xlabel(r'energy [GeV]')
	plt.ylabel(r'normalized counts')
	plt.xlim(0.,9.)
	save_name = str(E).replace(".","_")+"GeV_Air_energyScale.png"
	plt.savefig(save_name)
	plt.close()

	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	plt.bar(binsE_alu, counts_alu, width=binningE, color=color_alu)
	ax.legend(['Aluminium'], title=(str(E)+' GeV'), fontsize='small')
	plt.yscale('log')
	plt.ylim(1e-5, 3e-1)
	plt.xlabel(r'energy [GeV]')
	plt.ylabel(r'normalized counts')
	plt.xlim(0.,9.)
	save_name = str(E).replace(".","_")+"GeV_Aluminium_energyScale.png"
	plt.savefig(save_name)
	plt.close()

	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	plt.bar(binsE_tungsten, counts_tungsten, width=binningE, color=color_tungsten)
	ax.legend(['Tungsten'], title=(str(E)+' GeV'), fontsize='small')
	plt.yscale('log')
	plt.ylim(1e-5, 3e-1)
	plt.xlabel(r'energy [GeV]')
	plt.ylabel(r'normalized counts')
	plt.xlim(0.,9.)
	save_name = str(E).replace(".","_")+"GeV_Tungsten_energyScale.png"
	plt.savefig(save_name)
	plt.close()

	fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
	fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.17)
	bar1 = plt.bar(binsE_air, counts_air, width=binningE, color=color_air, alpha=0.5)
	bar2 = plt.bar(binsE_alu, counts_alu, width=binningE, color=color_alu, alpha=0.5)
	bar3 = plt.bar(binsE_tungsten, counts_tungsten, width=binningE, color=color_tungsten, alpha=0.5)
	ax.legend( (bar1,bar2,bar3), ('Air','Aluminium','Tungsten'), title=(str(E)+' GeV'), fontsize='small')
	plt.yscale('log')
	plt.ylim(1e-5, 3e-1)
	plt.xlabel(r'energy [GeV]')
	plt.ylabel(r'normalized counts')
	plt.xlim(0.,9.)
	save_name = str(E).replace(".","_")+"GeV_overlay_energyScale.png"
	plt.savefig(save_name)
	plt.close()
