#!/usr/bin/env python3
"""Uses the `nanograin` module to calculate and plot silver iron grain size results"""

# for running as a script when not installed
import sys 
sys.path.append("/Users/brandon/Documents/Projects/grain-boundary-code")

import numpy as np
from nanograin.system import System
from nanograin.plot import plot_energy_vs_x_gb_for_d, plot_grain_size_vs_temperature_for_x_overall, plot_grain_size_vs_temperature_for_h_mix, plot_solubility_chart

#full_solute_list = ['H', 'Li', 'Be', 'B', 'C', 'N', 'Na', 'Mg', 'Al', 'Si', 'P', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U', 'Pu']"""

path = '/Users/brandon/Downloads/ThermodynamicStabilityModel/python/'

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
plot_energy_vs_x_gb_for_d(system, 0.03, 550, np.array([10, 15, 23.1, 30, 50, 1e12]), filename='fezr-energy-vs-grain-boundary-compositon.eps')
#again change the properties wrt Mark's code
system.h_mix = -24*1e3
plot_grain_size_vs_temperature_for_x_overall(system, np.arange(300-273,500-273, 20), np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05]), filename='fezr-grain-size-vs-temperature.eps', plot_inverse=True, inverse_filename='inverse-grain-size-vs-temperature.eps')
plot_grain_size_vs_temperature_for_h_mix(system, 0.04, np.arange(300-273,1400-273, 20), np.array([0, -20, -24, -25, -26, -30]), filename='fezr-grain-size-vs-temperature-h-mix.eps')

# plot a stability map
solute_list = ['B', 'Li']#'Ag', 'Be', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Tl', 'Pb', 'Th', 'U']
systems = []
for sol in solute_list:
    systems.append(System.from_json(path+'elements.json', path+'enthalpy.json', 'Fe', sol))
plot_solubility_chart(systems, 25, 0.10, filename='fe-stability-map', label_points=True)