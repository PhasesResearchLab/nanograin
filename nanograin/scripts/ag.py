#!/usr/bin/env python3
"""Uses the `nanograin` module to calculate and plot silver alloy grain size results"""

# for running as a script when not installed
import sys 
sys.path.append("/Users/brandon/Documents/Projects/grain-boundary-code")

import os
import numpy as np
import matplotlib.pyplot as plt 
from nanograin.system import System, plot_solubility_chart


#full_solute_list = ['H', 'Li', 'Be', 'B', 'C', 'N', 'Na', 'Mg', 'Al', 'Si', 'P', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U', 'Pu']"""

solute_list = ['B', 'Li']#'Fe', 'Be', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Tl', 'Pb', 'Th', 'U']
path = '/Users/brandon/Downloads/ThermodynamicStabilityModel/python/'

"""
#FeZr Example
system = System.from_json(path+'elements.json', path+'enthalpy.json', 'Fe', 'Zr')
# reset the properties to match the paper
system.gamma_0 = 0.795
system.gamma_surf['Zr']=1.909
system.gamma_surf['Fe']=2.417
system.bulk_modulus['Zr']=89.8*1e9
system.shear_modulus['Fe']=81.6*1e9
system.atomic_volume = 0.0118
system.h_mix = -25*1e3
system.h_elastic = -108*1e3
system.sigma = 31217
system.molar_volume['Fe']=7.107*1e-6
#system.plot_energy_vs_x_gb_for_d(0.03, 550, np.array([10, 15, 23.1, 30, 50, 1e12]), filename='energy-vs-grain-boundary-compositon.eps')
#again change the properties wrt Mark's code
system.h_mix = -24*1e3
#system.plot_grain_size_vs_temperature_for_x_overall(np.arange(300-273,1400-273, 20), np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05]), filename='grain-size-vs-temperature.eps', plot_inverse=True, inverse_filename='inverse-grain-size-vs-temperature.eps')
system.plot_grain_size_vs_temperature_for_h_mix(0.04, np.arange(300-273,1400-273, 20), np.array([0, -20, -24, -25, -26, -30]), filename='grain-size-vs-temperature-h-mix.eps')
"""

systems = []
for sol in solute_list:
    systems.append(System.from_json(path+'elements.json', path+'enthalpy.json', 'Ag', sol))

system = systems[0]
system.plot_energy_vs_x_gb_for_d(0.0308, 500, np.array([10, 15, 25, 30, 50, 1e12]))
system.plot_grain_size_vs_temperature_for_x_overall(np.arange(500, 700, 25), np.array([0.05, 0.01]))
system.plot_grain_size_vs_temperature_for_h_mix(0.0308, np.arange(500, 700, 25), np.array([0, -30]))
plot_solubility_chart(systems, 25, 0.10)

#system = System.from_json(path+'elements.json', path+'enthalpy.json','Ag','B')
#system.plot_energy_vs_x_gb_for_d(0.046, 100+273, np.array([10, 15, 23.1, 30, 50, 1e12]))
# Ag-B solubility range is from 700-1200 at a maximium X_B ~ 0.030"""
