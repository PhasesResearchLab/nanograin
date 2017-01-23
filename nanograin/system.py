"""Defines a binary alloy system and routines to use a thermodynamic
model of the grain boundary energy to understand the stability of an alloy
with respect to different properties"""

# TODO: write a minimizer that takes n free variables and does a mapping.

import json
import itertools
import numpy as np
from scipy.constants import R, N_A
from scipy.optimize import brentq

class System:
    """Defines the properties of a binary solvent-solute system"""

    def __init__(self, solvent, solute, gamma_0, h_mix, h_elastic, t_melt,
                 atomic_volume, sigma, gamma_surf, molar_mass, molar_volume,
                 bulk_modulus, shear_modulus, z, lattice_parameter):
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
        self.z = z # coordination
        self.lattice_parameter = lattice_parameter

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
        with open(element_file, encoding='utf-8') as element_json_file:
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
        z = {solvent: z_values[solvent_data['Crystal Structure']], solute: z_values[solute_data['Crystal Structure']]}
        t_melt = {solvent: solvent_data['Melting point °C']+273, solute: solute_data['Melting point °C']+273}
        lattice_parameter = {solvent: solvent_data['Lattice Parameter (Å) -a-'], solute: solute_data['Lattice Parameter (Å) -a-']}
        with open(enthalpy_file) as enthalpy_json_file:
            enthalpy_data = json.load(enthalpy_json_file)
        h_mix = enthalpy_data[solvent][solute]*1e3
        h_elastic = -2*(np.abs(molar_volume[solvent]-molar_volume[solute]))**2*bulk_modulus[solute]*shear_modulus[solvent]/(3*bulk_modulus[solute]*molar_volume[solvent]+4*shear_modulus[solvent]*molar_volume[solute])
        atomic_volume = molar_volume[solvent]/N_A*1e27
        sigma = N_A*(atomic_volume*1e-27)**(2/3)

        return cls(solvent, solute, gamma_0, h_mix, h_elastic, t_melt[solvent], atomic_volume, sigma, gamma_surf, molar_mass, molar_volume, bulk_modulus, shear_modulus, z, lattice_parameter)

    @classmethod
    def from_csv(cls, file, solvent, solute):
        """Create a system from a csv-like using pandas"""
        pass

    def theoretical_density(self, x_solute):
        """Return the ratio of the theoretical density of the alloy to the solvent
        
        Args:
            x_solute (float): the mole fraction of solute in the system
        """
        solvent = self.solvent
        solute = self.solute
        M_solvent = self.molar_mass[solvent]
        M_solute = self.molar_mass[solute]
        V_m_solvent = self.molar_volume[solvent]
        V_m_solute = self.molar_volume[solute]
        density_pure_solvent = M_solvent/V_m_solvent
        x_solvent = 1-x_solute
        density_alloy = (M_solvent*x_solvent+M_solute*x_solute)/(V_m_solvent*x_solvent+V_m_solute*x_solute)
        return density_alloy/density_pure_solvent

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

    def calc_solid_solution_strengthening(self, concentration):
        """This from Courtney's Mechanical Behavior of Materials 2nd Edition. Pgs 186-196"""
        beta = 3  # constant
        G_solvent = self.shear_modulus[self.solvent]/10**9
        G_solute = self.shear_modulus[self.solute]/10**9
        a_solvent = self.lattice_parameter[self.solvent]
        a_solute = self.lattice_parameter[self.solute]
        epsilon_b = np.abs((a_solute - a_solvent) / a_solvent)
        epsilon_g = (G_solute - G_solvent) / G_solvent
        epsilon_prime_g = epsilon_g / (1 + 0.5 * np.abs(epsilon_g))
        epsilon_s = np.abs(epsilon_prime_g - beta * epsilon_b)
        strengthening = G_solvent * epsilon_s ** (1.5) * concentration ** (0.5) / 700
        return strengthening

    def calculate_norm_gb_energy(self, x_solute_gb, temperature, grain_size, x_solute_sys, error_check=True):
        """Calculates the normalized grain boundary energy

        Args:
            x_solute_gb (ndarray): solute concentration in the grain boundary in J/m^2
            temperature (ndarray): temperature in K
            grain_size (ndarray): grain size in J/m^2
            x_solute_sys (ndarray): solute concentration in the system J/m^2
        Kwargs:
            error_check (bool): whether to check for errors in the arguments. Small speed up.
            (~10% for a single calculation, ~40% for larger (10^5) arrays). Defaults to True.
        Returns:
            ndarray of grain boundary energies
        Raises:
            ValueError if error_check is True and the data is out of bounds
        """
        if error_check:
            System.error_check_x(x_solute_gb)
            System.error_check_x(x_solute_sys, bounds=(0, 0.5))
            System.error_check_temperature(temperature)
            System.error_check_grain_size(grain_size)

        x_solute_interior = (6*self.atomic_volume**(1/3)/grain_size*x_solute_gb-x_solute_sys)/(6*self.atomic_volume**(1/3)/grain_size - 1)
        gb_energy = 1 + (2*(x_solute_gb - x_solute_interior)/(self.gamma_0*self.sigma))*((self.gamma_surf[self.solute]-self.gamma_surf[self.solvent])/6*self.sigma - self.h_mix * (17/3*x_solute_gb - 6*x_solute_interior + 1/6) + self.h_elastic - R*temperature*np.log((x_solute_interior*(1-x_solute_gb))/((1-x_solute_interior)*x_solute_gb))) #pylint: disable=E1101
        return gb_energy

    @np.vectorize
    def optimize_grain_size(self, overall_composition, temperature, bounds=(1, 100)):
        """Optimize the grain size with fixed temperature and x_overall using the brentq algorithm

        Note: Because this is vectorized YOU MUST EXPLICITLY PASS SELF
        Args:
            temperature (ndarray): temperature in Celsius
            overall_composition (ndarray): overall composition of the solute
        Kwargs:
            bounds ((float, float)): two-tuple of floats for the lower and upper bounds of brentq algorithm
        Returns:
            ndarray of stable grain size
        """
        # first just calculate with a small range of x_solute_gbs, if the min is negative at a large
        #  grain size, >100, lets say we cannot converge
        x_solute_gbs = np.arange(0.01, 0.5, 0.001) # domain over which GB energies will be found. May need to reduce number
        min_energy = self._gb_energies_optimize_scipy(bounds[0], x_solute_gbs, temperature, overall_composition)
        max_energy = self._gb_energies_optimize_scipy(bounds[1], x_solute_gbs, temperature, overall_composition)
        if min_energy*max_energy < 0:
            return brentq(self._gb_energies_optimize_scipy, bounds[0], bounds[1], args=(x_solute_gbs, temperature, overall_composition), xtol=0.0001)
        else:
            return np.nan

    def _gb_energies_optimize_scipy(self, stable_grain_size, x_solute_gbs, temperature, overall_composition):
        grain_boundary_energies = self.calculate_norm_gb_energy(x_solute_gbs, temperature + 273, stable_grain_size, overall_composition)
        return np.nanmin(grain_boundary_energies)

    def solute_can_stabilize(self, grain_size, max_solute_composition, temperature=None):
        """ Returns whether the solute can be stabilizied within the max_solute composition

        Args:
            grain_size (float): target grain size (in nm)
            max_solute_composition (float): maximium allowable overall molar composition of solute
        KWargs:
            temperature (float): temperature to check for stability. Defaults to 60% of the melting temperature (in Celsius, but shouldn't be?)
        Returns:
            Tuple of type (bool, float) for if the solute can stabilize and the stable composition
        """
        if not temperature:
            temperature = (self.t_melt)*0.6
        x_solute_gbs = np.arange(0.01, 0.5, 0.001) # domain over which GB energies will be found. May need to reduce number
        grain_boundary_energies = np.zeros((x_solute_gbs.shape))
        overall_composition = 0.001
        delta = 0.0001
        while True:
            # calculate the grain_bondary_energies
            grain_boundary_energies = self.calculate_norm_gb_energy(x_solute_gbs, temperature+273, grain_size, overall_composition) # TODO: speedup with all remaining gbe[k] = NaN after first?
            min_energy = np.nanmin(grain_boundary_energies)
            # if minimum energy is smaller than 0, it can be stabilized
            if min_energy <= 0:
                return (True, overall_composition)
            elif overall_composition >= max_solute_composition:
                return (False, 0.001)
            else:
                overall_composition += delta

    def calculate_energy_for_xgb_d(self, overall_composition, temperature, grain_sizes,
                                  x_solute_gb=np.linspace(0, 0.5, 50)):
        """Calculate GB energies for x_GBs and grain sizes at fixed temperature and x_overall

        Args:
            overall_composition (float): solute composition overall
            temperature (float): temperature of grain in Celsius
            grain_sizes (ndarray): array of grain sizes, this will be the number of rows
        Kwargs:
            x_solute_gb (ndarray): domain of the plot. Defaults to 0 to 0.5 (while the solute is a solute)
        Returns:
            A 2d array of energies for rows of grain sizes and columns of grain boundary solute composition
        """
        return self.calculate_norm_gb_energy(x_solute_gb[np.newaxis], temperature+273, grain_sizes[:, np.newaxis], overall_composition)

    def calculate_grain_size_for_temperature_x_overall(self, temperatures, overall_compositions):
        """Calculate a 2d array stabilized grain size for temperatures and x_overalls

        Args:
            temperatures (ndarray): temperature domain of the plot in Celsius
            overall_compositions (ndarray): array of compositions, this will be the number of lines plotted
        Returns:
            A 2d array of stable grain sizes for rows of compositions and columns of temperatures
        """
        return self.optimize_grain_size(self, overall_compositions[:,np.newaxis], temperatures[np.newaxis])

    def calculate_grain_size_for_temperature_h_mix(self, x_overall, temperatures, h_mixes):
        """Calculate stabilized grain sizes for temperatures and mixing enthalpies with fixed x_overall

        Args:
            x_overall (float): overall solute composition
            temperatures (ndarray): temperature domain of the plot in Celsius
            h_mixes (ndarray): array of compositions, this will be the number of lines plotted
        Returns:
            A 2d array of stable grain sizes for rows of h_mixes and columns of temperatures 
        """
        # since we need to change a self property, this cannot be vectorized completely.
        system_h_mix = self.h_mix
        grain_sizes = np.zeros((len(h_mixes), len(temperatures))) # create an empty space for grain sizes
        for i, h_mix in enumerate(h_mixes):
            self.h_mix = h_mix*1000
            grain_sizes[i] = self.optimize_grain_size(self, x_overall, temperatures[np.newaxis])
        self.h_mix = system_h_mix
        return grain_sizes

    @staticmethod
    def error_check_x(x, bounds=(0, 1)):
        """Raise an exception of mole fraction is out of bounds"""
        l = bounds[0] # lower bound
        u = bounds[1] # upper bound
        if isinstance(x, np.ndarray):
            if (x < l).any() or (x > u).any():
                raise ValueError('Solute concentration must be between x={} and x={}. Passed x={}.'.format(l, u, x))
        else:
            if x < l or x > u:
                raise ValueError('Solute concentration must be between x={} and x={}. Passed x={}.'.format(l, u, x))

    @staticmethod
    def error_check_temperature(temperature):
        if isinstance(temperature, np.ndarray):
            if (temperature < 0).any():
                raise ValueError('Grain boundary energy cannot be calculated for temperatures below zero. Passed {} K.'.format(temperature))
        else:
            if temperature < 0:
                raise ValueError('Grain boundary energy cannot be calculated for temperatures below zero. Passed {} K.'.format(temperature))

    @staticmethod
    def error_check_grain_size(d):
        if isinstance(d, np.ndarray):
            if (d <= 0).any():
                raise ValueError('Cannot calculate grain boundary energy for zero or negative grain sizes. Passed d={} nm.'.format(d))
        else:
            if d <= 0:
                raise ValueError('Cannot calculate grain boundary energy for zero or negative grain sizes. Passed d={} nm.'.format(d))


class TernarySystem():
    """Creates a ternary systems based on two non-interacting binaries."""
    def __init__(self, ab, ac):
        """Takes two systems A-B and A-C."""
        self.ab = ab
        self.ac = ac

    @classmethod
    def from_json(cls, element_file, enthalpy_file, solvent, solute_b, solute_c):
        ab = System.from_json(element_file, enthalpy_file, solvent, solute_b)
        ac = System.from_json(element_file, enthalpy_file, solvent, solute_c)
        return cls(ab, ac)

    def calculate_norm_gb_energy(self, x_b_gb, x_c_gb, x_b_sys, x_c_sys, temperature, grain_size, error_check=True):
        """Calculates the normalized grain boundary energy

        Args:
            x_solute_gb (ndarray): solute concentration in the grain boundary in J/m^2
            temperature (ndarray): temperature in K
            grain_size (ndarray): grain size in J/m^2
            x_solute_sys (ndarray): solute concentration in the system J/m^2
        Kwargs:
            error_check (bool): whether to check for errors in the arguments. Small speed up.
            (~10% for a single calculation, ~40% for larger (10^5) arrays). Defaults to True.
        Returns:
            ndarray of grain boundary energies
        Raises:
            ValueError if error_check is True and the data is out of bounds
        """
        # check for errors in passed data
        if error_check:
            System.error_check_x(x_b_gb)
            System.error_check_x(x_c_gb)
            System.error_check_x(x_b_sys, bounds=(0, 0.5))
            System.error_check_x(x_c_sys, bounds=(0, 0.5))
            System.error_check_temperature(temperature)
            System.error_check_grain_size(grain_size)

        ab = self.ab
        ac = self.ac
        a = ab.solvent
        b = ab.solute
        c = ac.solute
        x_b_i = (6*ab.atomic_volume**(1/3)/grain_size*x_b_gb-x_b_sys)/(6*ab.atomic_volume**(1/3)/grain_size-1)
        x_c_i = (6*ac.atomic_volume**(1/3)/grain_size*x_c_gb-x_c_sys)/(6*ac.atomic_volume**(1/3)/grain_size-1)
        # energy contributions
        excess_gb_solute = ((x_b_gb - x_b_i) / (ab.gamma_0 * ab.sigma) + (x_c_gb - x_c_i) / (ac.gamma_0 * ac.sigma))/2 # ab+ac
        surface_energy_diff = (((ab.gamma_surf[b] - ab.gamma_surf[a]) * ab.sigma)+\
                               ((ac.gamma_surf[c] - ac.gamma_surf[a]) * ac.sigma))/2 # ab+ac
        mixing_energy = ((ab.h_mix *(17 / 3 * x_b_gb - 6 * x_b_i + 1 / 6))+ \
                                              (ac.h_mix * (17 / 3 * x_c_gb - 6 * x_c_i + 1 / 6)))/2 # ab+ac
        elastic_num = ((-2*(np.abs(ab.molar_volume[a]-ab.molar_volume[b]))**2*ab.bulk_modulus[b]*ab.shear_modulus[a])-\
                       (2*(np.abs(ac.molar_volume[a]-ac.molar_volume[c]))**2*ac.bulk_modulus[c]*ac.shear_modulus[c]))/2 # ab+ac
        elastic_denom = ((3*ab.bulk_modulus[b] * ab.molar_volume[a] + 4 * ab.shear_modulus[a]*ab.molar_volume[b])+\
                         (3*ac.bulk_modulus[c] * ac.molar_volume[a] + 4 * ac.shear_modulus[a]*ac.molar_volume[c])) # ab+ac
        elastic_energy = elastic_num/elastic_denom # (ab+ac)/(ab+ac)
        entropy = np.log((x_b_i*(1-x_b_gb)+x_c_i*(1-x_c_gb)) /((1-x_b_i)*x_b_gb+(1-x_c_i)*x_c_gb)) #(ab+bc)/(ab+bc)

        gb_energy = 1 + 2*excess_gb_solute * ( \
            surface_energy_diff/6 - mixing_energy + elastic_energy - R*temperature*entropy)

        return gb_energy
