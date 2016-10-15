"""Defines a binary alloy system and routines to use a thermodynamic
model of the grain boundary energy to understand the stability of an alloy
with respect to different properties"""

import json
import itertools
import numpy as np

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
        # check for errors in passed data
        if x_solute_gb < 0 or x_solute_gb > 1:
            raise ValueError('Solute grain boundary concentration must be between x=0 and x=1. Passed x_gb={}.'.format(x_solute_gb))
        if temperature < 0:
            raise ValueError('Grain boundary energy cannot be calculated for temperatures below zero. Passed {} K.'.format(temperature))
        if grain_size <= 0:
            raise ValueError('Cannot calculate grain boundary energy for zero or negative grain sizes. Passed d={} nm.'.format(grain_size))
        if x_solute_sys < 0 or x_solute_sys > 0.5:
            raise ValueError('System solute concentrations must be between 0 and 0.5. Passed x_sys={}.'.format(x_solute_sys))

        x_solute_interior = (6*self.atomic_volume**(1/3)/grain_size*x_solute_gb-x_solute_sys)/(6*self.atomic_volume**(1/3)/grain_size - 1)
        gb_energy = 1 + (2*(x_solute_gb - x_solute_interior)/(self.gamma_0*self.sigma))*((self.gamma_surf[self.solute]-self.gamma_surf[self.solvent])/6*self.sigma - self.h_mix * (17/3*x_solute_gb - 6*x_solute_interior + 1/6) + self.h_elastic - System.R*temperature*np.log((x_solute_interior*(1-x_solute_gb))/((1-x_solute_interior)*x_solute_gb))) #pylint: disable=E1101
        return gb_energy

    def optimize_grain_size(self, overall_composition, temperature):
        """Optimize the grain size with fixed temperature and x_overall using a bisection method

        Args:
            temperature (float): temperature in Celsius
            overall_composition (float): overall composition of the solute
        """
        x_solute_gbs = np.arange(0.01, 0.5, 0.001) # domain over which GB energies will be found. May need to reduce number
        grain_boundary_energies = np.zeros((x_solute_gbs.shape))
        stable_grain_size = 1 # initial guess for grain size
        delta = 10.0 # initial change in grain size to converge
        tolerance = 0.00001 # stopping accuracy of stable_grain_size
        while True:
            # calculate the grain_bondary_energies
            for k, x_solute_gb in enumerate(x_solute_gbs):
                grain_boundary_energies[k] = self.calculate_norm_gb_energy(x_solute_gb, temperature+273, stable_grain_size, overall_composition) # TODO: speedup with all remaining gbe[k] = NaN after first?
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
            for k, x_solute_gb in enumerate(x_solute_gbs):
                grain_boundary_energies[k] = self.calculate_norm_gb_energy(x_solute_gb, temperature+273, grain_size, overall_composition) # TODO: speedup with all remaining gbe[k] = NaN after first?
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
        norm_gb_energy = np.ones((len(grain_sizes), len(x_solute_gb)))
        for i, j in itertools.product(range(len(grain_sizes)), range(len(x_solute_gb))):
            norm_gb_energy[i][j] = self.calculate_norm_gb_energy(x_solute_gb[j], temperature+273, grain_sizes[i], overall_composition)
        return norm_gb_energy
        
    def calculate_grain_size_for_temperature_x_overall(self, temperatures, overall_compositions):
        """Calculate a 2d array stablized grain size for temperatures and x_overalls

        Args:
            temperatures (ndarray): temperature domain of the plot in Celsius
            overall_compositions (ndarray): array of compositions, this will be the number of lines plotted
        Returns:
            A 2d array of stable grain sizes for rows of compositions and columns of temperatures
        """
        grain_sizes = np.zeros((len(overall_compositions), len(temperatures))) # create an empty space for grain sizes
        for i, overall_composition in enumerate(overall_compositions):
            for j, temperature in enumerate(temperatures):
                grain_sizes[i][j] = self.optimize_grain_size(overall_composition, temperature) # can be parallelized
        return grain_sizes

    def calculate_grain_size_for_temperature_h_mix(self, x_overall, temperatures, h_mixes):
        """Calculate stablized grain sizes for temperatures and mixing enthalpies with fixed x_overall

        Args:
            x_overall (float): overall solute composition
            temperatures (ndarray): temperature domain of the plot in Celsius
            h_mixes (ndarray): array of compositions, this will be the number of lines plotted
        Returns:
            A 2d array of stable grain sizes for rows of h_mixes and columns of temperatures 
        """
        system_h_mix = self.h_mix
        grain_sizes = np.zeros((len(h_mixes), len(temperatures))) # create an empty space for grain sizes
        for i, h_mix in enumerate(h_mixes):
            self.h_mix = h_mix*1000
            for j, temperature in enumerate(temperatures):
                grain_sizes[i][j] = self.optimize_grain_size(x_overall, temperature) # can be parallelized
        self.h_mix = system_h_mix
        return grain_sizes