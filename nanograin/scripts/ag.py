#!/usr/bin/env python3
"""Uses the `nanograin` module to calculate and plot silver alloy grain size results"""

# for running as a script when not installed
import sys 
sys.path.append("/Users/brandon/Documents/Projects/grain-boundary-code")

import numpy as np
from nanograin.system import System
from nanograin.plot import plot_energy_vs_x_gb_for_d, plot_grain_size_vs_temperature_for_x_overall, plot_grain_size_vs_temperature_for_h_mix, plot_solubility_chart

#full_solute_list = ['H', 'Li', 'Be', 'B', 'C', 'N', 'Na', 'Mg', 'Al', 'Si', 'P', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U', 'Pu']"""

solute_list = ['B', 'Li']#'Fe', 'Be', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Tl', 'Pb', 'Th', 'U']
path = '/Users/brandon/Downloads/ThermodynamicStabilityModel/python/'

systems = []
for sol in solute_list:
    systems.append(System.from_json(path+'elements.json', path+'enthalpy.json', 'Ag', sol))

system = systems[0]
plot_energy_vs_x_gb_for_d(system, 0.0308, 500, np.array([10, 15, 25, 30, 50, 1e12]))
plot_grain_size_vs_temperature_for_x_overall(system, np.arange(500, 700, 25), np.array([0.05, 0.01]))
plot_grain_size_vs_temperature_for_h_mix(system, 0.0308, np.arange(500, 700, 25), np.array([0, -30]))
plot_solubility_chart(systems, 25, 0.10)

#system = System.from_json(path+'elements.json', path+'enthalpy.json','Ag','B')
#system.plot_energy_vs_x_gb_for_d(0.046, 100+273, np.array([10, 15, 23.1, 30, 50, 1e12]))
# Ag-B solubility range is from 700-1200 at a maximium X_B ~ 0.030"""
