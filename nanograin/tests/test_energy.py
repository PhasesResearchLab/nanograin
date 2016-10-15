"""Tests if the normalized grain boundary function is calculating the correct values

Uses the values calculated from this code as compared with the MATLAB code from
Mark Tschopp in the FeZr system. The MATLAB code answers are commented out and
should show that they compare well with the calculated answers here. The reason
the answers here are used is because much higher precision can be tested for if
the calculated values are used over the oroginal values.  In the future, the
FeZr system should be made static (and not loaded from file) for both packaging
and consistency reasons. Changing the FeZr data files would cause this test
module to probably fail.

Note that these are currently checking for precision to 4 decimals. Precision
is checking more of the absolute value. Is zero really zero, etc. Precision is
not checking whether 1.5e-6 is the same as 6e-6, up a precision of 5 decimal
places, they are the same. Here, precision makes more sense to use than
accuracy, because these are energies and the order of magnitude of the values
and the relative magnitude of the calculated values is much more important than
what the values are themselves. Accuracy is too hard for computers dealing with
very small and very large numbers to get correct between implementations because
of overflow and underflow issues."""

import numpy as np
import numpy.testing as npt
import pytest
from nanograin.system import System

path = '/Users/brandon/Downloads/ThermodynamicStabilityModel/python/'
fezr_system = System.from_json(path+'elements.json', path+'enthalpy.json', 'Fe', 'Zr')

def test_grain_boundary_energies_over_temperatures():
    """Tests grain boundary energy calculations against those calculated in MATLAB within a tolerance"""
    energy = fezr_system.calculate_norm_gb_energy
    npt.assert_almost_equal(energy(0.27, 0, 23.1, 0.03), -0.43856361611, decimal=7) # -4.385423e-1
    npt.assert_almost_equal(energy(0.27, 273, 23.1, 0.03), -0.293085746222, decimal=7) # -2.930663e-1
    npt.assert_almost_equal(energy(0.27, 573, 23.1, 0.03), -0.133219955136, decimal=7) # -1.332025e-1
    npt.assert_almost_equal(energy(0.27, 773, 23.1, 0.03), -0.0266427610793, decimal=7) # -2.66266e-2
    npt.assert_almost_equal(energy(0.27, 550+273, 23.1, 0.03), 1.53743504527e-06, decimal=7) #zero point # 1.738447e-5
    npt.assert_almost_equal(energy(0.27, 973, 23.1, 0.03), 0.079934432978, decimal=7) # 7.994929e-2
    npt.assert_almost_equal(energy(0.27, 1573, 23.1, 0.03), 0.39966601515, decimal=7) # 3.996769e-1
    npt.assert_almost_equal(energy(0.27, 10000, 23.1, 0.03), 4.89029608675, decimal=7) # 4.890251

def test_grain_boundary_energies_over_grain_boundary_solute_concentrations():
    """Tests grain boundary energy calculations against those calculated in MATLAB within a tolerance"""
    energy = fezr_system.calculate_norm_gb_energy
    npt.assert_almost_equal(energy(0.01, 823, 25, 0.03), 1.19781693583, decimal=7) # 1.197814
    npt.assert_almost_equal(energy(0.10, 823, 25, 0.03), 0.489441298658, decimal=7) # 4.894489e-1
    npt.assert_almost_equal(energy(0.30, 823, 25, 0.03), 0.00573341857976, decimal=7) # 5.749406e-3

def test_grain_boundary_energies_over_solute_concentrations():
    """Tests grain boundary energy calculations against those calculated in MATLAB within a tolerance"""
    energy = fezr_system.calculate_norm_gb_energy
    npt.assert_almost_equal(energy(0.27, 823, 25, 0.10), -0.0463451775354, decimal=7) # -4.632963e-2
    npt.assert_almost_equal(energy(0.27, 823, 25, 0.25), 0.823572510452, decimal=7) # 8.235751e-1
    npt.assert_almost_equal(energy(0.27, 823, 25, 0.49), 3.7772175566, decimal=7) # 3.777177
    npt.assert_almost_equal(energy(0.27, 823, 25, 0.50), 3.93934911851, decimal=7)# 3.939396

def test_grain_boundary_energies_over_grain_sizes():
    """Tests grain boundary energy calculations against those calculated in MATLAB within a tolerance"""
    energy = fezr_system.calculate_norm_gb_energy
    npt.assert_almost_equal(energy(0.27, 823, 1, 0.03), 13.0068219814, decimal=7) # 1.300604e1
    npt.assert_almost_equal(energy(0.27, 823, 20, 0.03), 0.0236545668524, decimal=7) # 2.367054e-2
    npt.assert_almost_equal(energy(0.27, 823, 100, 0.03), -0.0652978075393, decimal=7) # -6.528187e-2
    npt.assert_almost_equal(energy(0.27, 823, 1000, 0.03), -0.0748316221538, decimal=7) # -7.481561e-2

def test_grain_boundary_energies_with_non_real_parts_are_NaN():
    """Tests that calculated energies that return complex results in MATLAB are NaN"""
    energy = fezr_system.calculate_norm_gb_energy
    assert np.isnan(energy(0.27, 823, 5, 0.03)) # -2.36631e-1-5.58125e-1i 
    assert np.isnan(energy(0.27, 823, 25, 0.0)) #-1.8982e-2-4.8277e-2i
    assert np.isnan(energy(0.27, 823, 25, 0.01)) #1.5060e-1-4.6489e-1 i 
    assert np.isnan(energy(0.90, 823, 25, 0.03)) #5.67877-1.55560i
    assert np.isnan(energy(0.99, 823, 25, 0.03)) #8.43130-1.71652i

def test_grain_boundary_energy_at_1_x_gb_solute_gives_Inf():
    """Tests that calculating the energy for zero x_gb_solute is Inf"""
    assert fezr_system.calculate_norm_gb_energy(1, 823, 25, 0.03) == np.Inf #should this be? or NaN?

def test_grain_boundary_energy_zero_x_gb_solute_raises_zero_division_error():
    """Tests that calculating the energy for zero x_gb_solute raises a ZeroDivisionError"""
    with pytest.raises(ZeroDivisionError):
         fezr_system.calculate_norm_gb_energy(0, 823, 25, 0.03)

def test_grain_boundary_energy_zero_x_gb_solute_and_0_temperature_raises_zero_division_error():
    """Tests that calculating the energy for zero x_gb_solute and temperature raises a ZeroDivisionError"""
    with pytest.raises(ZeroDivisionError):
         fezr_system.calculate_norm_gb_energy(0, 0, 25, 0.03)

def test_grain_boundary_energy_raises_error_for_negative_temperature():
    """Tests that the temperature cannot be negative"""
    with pytest.raises(ValueError):
        fezr_system.calculate_norm_gb_energy(0.27, -1, 23.1, 0.03) 

def test_grain_boundary_energy_raises_error_for_negative_grain_boundary_concentrations():
    """Tests that the grain boundary concentration cannot be negative"""
    with pytest.raises(ValueError):
        fezr_system.calculate_norm_gb_energy(-1, 823, 23.1, 0.03) 

def test_grain_boundary_energy_raises_error_for_large_grain_boundary_concentrations():
    """Tests that the grain boundary concentration cannot be greater than 1"""
    with pytest.raises(ValueError):
        fezr_system.calculate_norm_gb_energy(1.01, 823, 23.1, 0.03) 
    with pytest.raises(ValueError):
        fezr_system.calculate_norm_gb_energy(10, 823, 23.1, 0.03) 

def test_grain_boundary_energy_raises_error_for_negative_solute_concentrations():
    """Test that the solute concentration can not be negative"""
    with pytest.raises(ValueError):
        fezr_system.calculate_norm_gb_energy(0.27, 823, 23.1, -1) 

def test_grain_boundary_energy_raises_error_for_solute_concentrations_above_solvent():
    """Test that the solute concentration is not above 0.5"""
    with pytest.raises(ValueError):
        fezr_system.calculate_norm_gb_energy(0.27, 823, 23.1, 0.51) 
    with pytest.raises(ValueError):
        fezr_system.calculate_norm_gb_energy(0.27, 823, 23.1, 1) 

def test_grain_boundary_energy_raises_error_for_negative_grain_sizes():
    """Tests that the grain boundary concentration cannot be negative"""
    with pytest.raises(ValueError):
        fezr_system.calculate_norm_gb_energy(0.27, 823, 0, 0.03) 
    with pytest.raises(ValueError):
        fezr_system.calculate_norm_gb_energy(0.27, 823, -1, 0.03) 
