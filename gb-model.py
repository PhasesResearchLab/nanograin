#!/usr/bin/env python3

"""Module defines a binary alloy system and routines to use a thermodynamic
model of the grain boundary energy to understand the stability of an alloy
with respect to different properties"""

import json
import os
import itertools
import numpy as np
import matplotlib as mpl
mpl.rc('font', family='serif')
#mpl.rc('legend',  loc=0) # fontsize=10,
import matplotlib.pyplot as plt

# todo: use pint or another unit manager
class System:
    """Defines the properties of a binary solvent-solute system"""

    R = 8.3144598484848484 # gas constant, J/mol/K
    N_A = 6.022140857747474e23 # Avagadro's number, 1/mol

    def __init__(self, solvent, solute, gamma_0, h_mix, h_elastic, t_melt,
                 atomic_volume, sigma, gamma_surf, molar_mass, molar_volume,
                 bulk_modulus, shear_modulus, z):
        # direct properties
        self.solvent = solvent
        self.solute = solute
        self.gamma_0 = gamma_0
        self.h_mix = h_mix
        self.h_elastic = h_elastic
        self.t_melt = t_melt
        self.atomic_volume = atomic_volume
        self.sigma = sigma # grain boundary area
        # dictionaries
        self.gamma_surf = gamma_surf
        self.molar_mass = molar_mass
        self.molar_volume = molar_volume
        self.bulk_modulus = bulk_modulus
        self.shear_modulus = shear_modulus
        self.z = z

    @classmethod
    def from_json(cls, element_file, enthalpy_file, solvent, solute):
        """Creates a system from a JSON file

        The element symbols should be the top level keys of the JSON file,
        with the values of the keys being a dictionary of different properties
        or data.

        Args:
            file (str): filename of a JSON file containing periodic table data
            solvent (str): chemical symbol of the solvent
            solute (str): chemical symbol of the solute
        """
        with open(element_file) as element_json_file:
            element_data = json.load(element_json_file)
        # solvent and solute property dicts
        solvent_data = element_data[solvent]
        solute_data = element_data[solute]

        gamma_surf = {solvent: solvent_data['Surface Energy (J/mol2)'], solute: solute_data['Surface Energy (J/mol2)']}
        gamma_0 = gamma_surf[solvent]/3
        bulk_modulus = {solvent: solvent_data['Bulk Modulus, K (GPa)']*10**9, solute: solute_data['Bulk Modulus, K (GPa)']*10**9}
        shear_modulus = {solvent: solvent_data['Shear Modulus, G (GPa)']*10**9, solute: solute_data['Shear Modulus, G (GPa)']*10**9}
        molar_mass = {solvent: solvent_data['Atomic mass (g/mol)'], solute: solute_data['Atomic mass (g/mol)']}
        molar_volume = {solvent: solvent_data['Atomic Volume (cm3/mol)']/100**3, solute: solute_data['Atomic Volume (cm3/mol)']/100**3}
        z_values = {'fcc': 12, 'bcc': 8, 'hcp': 12, 'ortho': 6, 'tetrag': 6, 'diamond': 4} #check hex
        z = {solvent: z_values[solvent_data['Crystal Structure']], solute: z_values[solute_data['Crystal Structure']]} # what is this? depends on structure
        t_melt = {solvent: solvent_data['Melting point °C']+273, solute: solute_data['Melting point °C']+273}

        with open(enthalpy_file) as enthalpy_json_file:
            enthalpy_data = json.load(enthalpy_json_file)
        h_mix = enthalpy_data[solvent][solute]*1e3
        h_elastic = -2*(np.abs(molar_volume[solvent]-molar_volume[solute]))**2*bulk_modulus[solute]*shear_modulus[solvent]/(3*bulk_modulus[solute]*molar_volume[solvent]+4*shear_modulus[solvent]*molar_volume[solute])
        atomic_volume = molar_volume[solvent]/System.N_A*1e27
        sigma = System.N_A*(atomic_volume*1e-27)**(2/3)

        return cls(solvent, solute, gamma_0, h_mix, h_elastic, t_melt[solvent], atomic_volume, sigma, gamma_surf, molar_mass, molar_volume, bulk_modulus, shear_modulus, z)

    @classmethod
    def from_csv(cls, file, solvent, solute):
        """Create a system from a csv-like using pandas"""
        pass

    def max_x_solute_sys(self, density_difference=0.10):
        """Return the maximum solute concentration relative to deviation from solvent density

        Args:
            density_difference (float): maximum deviation from the solvent density

        Only works for binary systems currently. The base formula is
        d = (M_A*X_A+M_B*X_B)/(V_m_A*X_A+V_m_B*X_B)
        """
        solvent = self.solvent
        solute = self.solute
        M_solvent = self.molar_mass[solvent]
        M_solute = self.molar_mass[solute]
        V_m_solvent = self.molar_volume[solvent]
        V_m_solute = self.molar_volume[solute]
        density_silver = M_solvent/V_m_solvent
        upper_target_density = (1+density_difference)*density_silver
        lower_target_density = (1-density_difference)*density_silver
        x_solute_upper = (upper_target_density*V_m_solvent-M_solvent)/(upper_target_density*(V_m_solvent-V_m_solute)+M_solute-M_solvent)
        x_solute_lower = (lower_target_density*V_m_solvent-M_solvent)/(lower_target_density*(V_m_solvent-V_m_solute)+M_solute-M_solvent)
        if x_solute_upper >= 0 and x_solute_upper <= 1:
            return x_solute_upper
        elif x_solute_lower >= 0 and x_solute_lower <= 1:
            return x_solute_lower
        else:
            raise ValueError('The solute fractions of {} for a {:.2f}{} density difference from {} are out of range'.format(self.solute, density_difference*100, '%', self.solvent))

    def calculate_norm_gb_energy(self, x_solute_gb, temperature, grain_size, x_solute_sys):
        """Calculates the normalized grain boundary energy

        Args:
            x_solute_gb (float): solute concentration in the grain boundary in J/m^2
            temperature (float): temperature in K
            grain_size (float): grain size in J/m^2
            x_solute_sys (float): solute concentration in the system J/m^2
            system (dict): properties of the system

        TODO: it would be neat if this made an N-dimensional array if ndarrays
        were passed for the arguments. All of the plots would just be slices
        of this array. Variables could be optionally fixed.
        """
        x_solute_interior = (6*self.atomic_volume**(1/3)/grain_size*x_solute_gb-x_solute_sys)/(6*self.atomic_volume**(1/3)/grain_size - 1)
        gb_energy = 1 + (2*(x_solute_gb - x_solute_interior)/(self.gamma_0*self.sigma))*((self.gamma_surf[self.solute]-self.gamma_surf[self.solvent])/6*self.sigma - self.h_mix * (17/3*x_solute_gb - 6*x_solute_interior + 1/6) + self.h_elastic - System.R*temperature*np.log((x_solute_interior*(1-x_solute_gb))/((1-x_solute_interior)*x_solute_gb)))
        return gb_energy

    def optimize_grain_size(self, overall_composition, temperature):
        """Optimize the grain size with fixed temperature and x_overall using a bisection method

        Args:
            temperature (float): temperature in Celsius
            overall_composition (float): overall composition of the solute
        """
        # initialize values
        #x_solute_gb = np.linspace(0.001, 0.499, 1000) # domain over which GB energies will be found. May need to reduce number
        x_solute_gbs = np.arange(0.01, 0.5, 0.001) # domain over which GB energies will be found. May need to reduce number
        grain_boundary_energies = np.zeros((x_solute_gbs.shape))
        stable_grain_size = 1 # initial guess for grain size
        delta = 10.0 # initial change in grain size to converge
        tolerance = 0.00001 # stopping accuracy of stable_grain_size
        while True:
            # calculate the grain_bondary_energies
            for k, x_solute_gb in enumerate(x_solute_gbs):
                grain_boundary_energies[k] = self.calculate_norm_gb_energy(x_solute_gb, temperature+273, stable_grain_size, overall_composition)
            # skipping step to eliminate non-physical x_solute_interior values
            # find the minimum energy
            min_energy = np.nanmin(grain_boundary_energies)
            # if minimum energy is greater than 0+tolerance, increase the grain size and the reverse
            if min_energy > 0:
                if delta < 0:
                    delta = -1*delta/2
                stable_grain_size += delta
            elif min_energy < 0:
                if delta > 0:
                    delta = -1*delta/2
                stable_grain_size += delta
            if np.abs(delta) < tolerance:
                return stable_grain_size
            elif stable_grain_size > 250:
                return np.nan # assume destabilized

    def solute_can_stablize(self, grain_size, max_solute_composition, temperature=None):
        """ Returns whether the solute can be stablizied within the max_solute composition

        Args:
            grain_size (float): target grain size (in nm)
            max_solute_composition (float): maximium allowable overall molar composition of solute
        KWargs:
            temperature (float): temperature to check for stability. Defaults to 60% of the melting temperature (in Celsius, but shouldn't be?)
        """
        if not temperature:
            temperature = (self.t_melt)*0.6
        x_solute_gbs = np.arange(0.01, 0.5, 0.001) # domain over which GB energies will be found. May need to reduce number
        grain_boundary_energies = np.zeros((x_solute_gbs.shape))
        overall_composition = 0.001
        delta = 0.0001
        while True:
            # calculate the grain_bondary_energies
            for k, x_solute_gb in enumerate(x_solute_gbs):
                grain_boundary_energies[k] = self.calculate_norm_gb_energy(x_solute_gb, temperature+273, grain_size, overall_composition)
            min_energy = np.nanmin(grain_boundary_energies)
            # if minimum energy is smaller than 0, it can be stabilized
            if min_energy <= 0:
                return (True, overall_composition)
            elif overall_composition >= max_solute_composition:
                return (False, 0.001)
            else:
                overall_composition += delta

    def plot_energy_vs_x_gb_for_d(self, overall_composition, temperature, grain_sizes,
                                  x_solute_gb=np.linspace(0, 0.5, 50), filename=None,
                                  markers=['o', 'v', 'd', '^', '<', '>', 's', '*', 'x', '+', '1', '2']):
        """Plot a figure of GB energy vs. x_GB for different grain sizes at fixed temperature and x_overall

        Args:
            overall_composition (float): solute composition overall
            temperature (float): temperature of grain in Celsius
            grain_sizes (ndarray): array of grain sizes, this will be the number of lines plotted
        Kwargs:
            x_solute_gb (ndarray): domain of the plot. Defaults to 0 to 0.5 (while the solute is a solute)
            filename (str): filename of the saved figure with extension. If no filename is provided, figure will not be saved.
            markers ([str]): list of matplotlib markers that can be overridden
        """
        norm_gb_energy = np.ones((len(grain_sizes), len(x_solute_gb)))
        for i, j in itertools.product(range(len(grain_sizes)), range(len(x_solute_gb))):
            norm_gb_energy[i][j] = self.calculate_norm_gb_energy(x_solute_gb[j], temperature+273, grain_sizes[i], overall_composition)
        # Plot
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
        for i, grain_size in enumerate(grain_sizes):
            plt.plot(x_solute_gb, norm_gb_energy[:][i], marker=markers[i], markersize=5, linestyle='None', label='d = {} nm'.format(str(grain_size)))
        ax.axhline(0, color='k')
        plt.legend(frameon=False, loc=0)
        plt.title(r'Grain Boundary Energy of {}-{}'.format(self.solvent, self.solute))
        plt.xlabel(self.solute + r' grain boundary mole fraction, $X^\mathrm{GB}_{\mathrm{'+self.solute+r'}}$', size=15)
        plt.ylabel(r'Normalized grain boundary energy, $\gamma/\gamma_0$', size=15)
        if filename:
            fig.savefig(filename)
            plt.close(fig)

    def plot_grain_size_vs_temperature_for_x_overall(self, temperatures, overall_compositions,
                                                     filename=None, plot_inverse=True,
                                                     inverse_filename=None,
                                                     markers=['o', 'v', 'd', '^', '<', '>', 's', '*', 'x', '+', '1', '2']):
        """Plot a figure of stablized grain size vs. temperature for different x_overall at fixed x_GB

        Args:
            temperatures (ndarray): temperature domain of the plot in Celsius
            overall_compositions (ndarray): array of compositions, this will be the number of lines plotted
        Kwargs:
            filename (str): filename of the saved figure with extension. If no filename is provided, figure will not be saved.
            plot_inverse (bool): whether to plot the inverse grain size plot
            inverse_filename (str): filename for an inverse plot. Will not work if plot_inverse is False. May change in future to effectively make plot_inverse true
            markers ([str]): list of matplotlib markers that can be overridden
        """
        grain_sizes = np.zeros((len(overall_compositions), len(temperatures))) # create an empty space for grain sizes
        for i, overall_composition in enumerate(overall_compositions):
            for j, temperature in enumerate(temperatures):
                grain_sizes[i][j] = self.optimize_grain_size(overall_composition, temperature) # can be parallelized
        # Plot
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
        for i, overall_composition in enumerate(overall_compositions):
            plt.plot(temperatures, grain_sizes[i][:], marker=markers[i], markersize=5, linestyle='--', label='$X_0 = $ {}'.format(overall_composition))
        ax.axhline(0, color='k')
        plt.legend(frameon=False, loc=0)
        plt.title(r'Stabilized Grain Sizes of {}-{}'.format(self.solvent, self.solute))
        plt.xlabel(r'Temperature, $T$ $(C)$', size=15)
        plt.ylabel(r'Stabilized grain size, $d_m$ $(nm)$', size=15)
        if filename:
            fig.savefig(filename)
            plt.close(fig)
        
        if plot_inverse:
            fig, ax = plt.subplots()
            fig.set_size_inches(6, 6)
            for i, overall_composition in enumerate(overall_compositions):
                plt.plot(temperatures, 1/grain_sizes[i][:], marker=markers[i], markersize=5, linestyle='--', label='$X_0 = $ {}'.format(overall_composition))
            ax.axhline(0, color='k')
            plt.legend(frameon=False, loc=0)
            plt.title(r'Inverse Stabilized Grain Sizes of {}-{}'.format(system.solvent, system.solute))
            plt.xlabel(r'Temperature, $T$ $(C)$', size=15)
            plt.ylabel(r'Inverse stabilized grain size, $1/d_m$ $(nm^{-1})$', size=15)
            if inverse_filename:
                fig.savefig(inverse_filename)
                plt.close(fig)

    def plot_grain_size_vs_temperature_for_h_mix(self, x_overall, temperatures,
                                                 h_mixes, filename=None,
                                                 markers=['o', 'v', 'd', '^', '<', '>', 's', '*', 'x', '+', '1', '2']):
        """Plot a figure of stablized grain size vs. temperature for different mixing enthalpies with fixed x_overall

        Args:
            x_overall (float): overall solute composition
            temperatures (ndarray): temperature domain of the plot in Celsius
            h_mixes (ndarray): array of compositions, this will be the number of lines plotted
        Kwargs:
            filename (str): filename of the saved figure with extension. If no filename is provided, figure will not be saved.
            markers ([str]): list of matplotlib markers that can be overridden
        """
        system_h_mix = self.h_mix

        grain_sizes = np.zeros((len(h_mixes), len(temperatures))) # create an empty space for grain sizes
        for i, h_mix in enumerate(h_mixes):
            self.h_mix = h_mix*1000
            for j, temperature in enumerate(temperatures):
                grain_sizes[i][j] = self.optimize_grain_size(x_overall, temperature) # can be parallelized
        # Plot
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
        for i, h_mix in enumerate(h_mixes):
            plt.plot(temperatures, grain_sizes[i][:], marker=markers[i], markersize=5, linestyle='--', label=r'$\Delta H_\mathrm{mix} = $ '+ str(h_mix)+ r' $\mathrm{kJ/mol}$')
        ax.axhline(0, color='k')
        plt.legend(frameon=False, loc=0, prop={'size':10})
        plt.title(r'Stabilized Grain Sizes of {}-{}'.format(system.solvent, system.solute))
        plt.xlabel(r'Temperature, $T$ $(C)$', size=15)
        plt.ylabel(r'Stabilized grain size, $d_m$ $(nm)$', size=15)
        if filename:
            fig.savefig(filename)
            plt.close(fig)

        self.h_mix = system_h_mix

def plot_solubility_chart(systems, grain_size, max_solute_composition, filename=None, label_points=False, interactive_plot=False, axis_limits=None, temperature=None):
    """Plot a solubility chart for the given systems and target grain size

    Args:
        systems ([System]): list of systems
        grain_size (float): target grain size in (nm)
        max_solute_composition (float): maximium allowable overall molar composition of solute in the solvent
    Kwargs:
            filename (str): filename of the saved figure with extension. If no filename is provided, figure will not be saved.
        axis_limits ([float]): matplotlib axis limits stye list, [x_min, x_max, y_min, y_max]

    """
    stable_list = []
    composition_list = []
    for system in systems:
        (is_stable, overall_composition) = system.solute_can_stablize(grain_size, max_solute_composition, temperature=temperature)
        stable_list.append(is_stable)
        composition_list.append(overall_composition)
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    if axis_limits:
        ax.axis(axis_limits)
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    composition_min = np.min(composition_list)
    composition_max = np.max(composition_list)
    for system, stable, composition in zip(systems, stable_list, composition_list):
        if stable:
            plt.plot(system.h_elastic/1000, system.h_mix/1000, marker='o', color='r', markersize=((composition-composition_min)/(composition_max-composition_min)+1)*4, linestyle='')
        else:
            plt.plot(system.h_elastic/1000, system.h_mix/1000, marker='o', color='k', markersize=((composition-composition_min)/(composition_max-composition_min)+1)*4, linestyle='')
        if label_points:
            plt.annotate('{}'.format(system.solute), xy=(system.h_elastic/1000, system.h_mix/1000), size=10, textcoords='data')
    plt.title(r'{} Solubility Map'.format(system.solvent))
    plt.xlabel(r'Enthalpy of mixing, $\Delta H_{\mathrm{mix}}$ $\mathrm{kJ/mol}$', size=15)
    plt.ylabel(r'Elastic enthalpy, $\Delta_{\mathrm{seg}}^\mathrm{{elastic}}$ $\mathrm{kJ/mol}$', size=15)
    if interactive_plot:
        plt.show()
    if filename:
        plt.gcf().subplots_adjust(left=0.2)
        fig.savefig(filename)
        plt.close(fig)


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

#solute_list = ['Zr', 'Th', 'Sc', 'W', 'Ti', 'B', 'C'] #, 'Al', 'Ni', 'W', 'Ti', 'Pd', 'B', 'C']
#solute_list = [H, Li, Be, B, C, N, Na, Mg, Al, Si, P, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Cs, Ba, La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Th, U, Pu]
#solute_list = ['H', 'Li', 'Be', 'B', 'C', 'N', 'Na', 'Mg', 'Al', 'Si', 'P', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U', 'Pu']"""


# they want a grain size of 25 nm.

solute_list = ['B', 'Li']#'Fe', 'Be', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Tl', 'Pb', 'Th', 'U']
systems = []
for sol in solute_list:
    systems.append(System.from_json(path+'elements.json', path+'enthalpy.json', 'Ag', sol))

system = systems[0]
os.system('mkdir {}/{}'.format(system.solvent, system.solute))
print('starting energy vs x gb')
system.plot_energy_vs_x_gb_for_d(0.0308, 500, np.array([10, 15, 25, 30, 50, 1e12]), filename='{}/{}/test-e-vs-x-gb.eps'.format(system.solvent, system.solute))
print('starting grain size vs t')
system.plot_grain_size_vs_temperature_for_x_overall(np.arange(500, 700, 25), np.array([0.05, 0.01]), filename='{}/{}/test-d-vs-t-x-overall.eps'.format(system.solvent, system.solute), plot_inverse=True, inverse_filename='{}/{}/inverse-grain-size-vs-temperature.eps'.format(system.solvent, system.solute))
print('starting grain size vs t for h_mix')
system.plot_grain_size_vs_temperature_for_h_mix(0.0308, np.arange(500, 700, 25), np.array([0, -30]), filename='{}/{}/test-d-vs-t-h-mix.eps'.format(system.solvent, system.solute))
plot_solubility_chart(systems, 25, 0.10, filename='test-solubility-chart.eps', label_points=True, axis_limits=[-120, 30, -50, 50], interactive_plot=True)

#system = System.from_json(path+'elements.json', path+'enthalpy.json','Ag','B')
#system.plot_energy_vs_x_gb_for_d(0.046, 100+273, np.array([10, 15, 23.1, 30, 50, 1e12]))
# Ag-B solubility range is from 700-1200 at a maximium X_B ~ 0.030"""
