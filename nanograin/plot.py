"""Provides routines for plotting results calculated for a System"""

import numpy as np
import matplotlib as mpl
mpl.rc('font', family='serif')
#mpl.rc('legend',  loc=0) # fontsize=10,
import matplotlib.pyplot as plt

def plot_energy_vs_x_gb_for_d(system, overall_composition, temperature, grain_sizes,
                                  x_solute_gb=np.linspace(0, 0.5, 50), filename=None,
                                  markers=['o', 'v', 'd', '^', '<', '>', 's', '*', 'x', '+', '1', '2']):
    """Plot a figure of GB energies for x_GBs and grain sizes at fixed temperature and x_overall

        Args:
            overall_composition (float): solute composition overall
            temperature (float): temperature of grain in Celsius
            grain_sizes (ndarray): array of grain sizes, this will be the number of rows
        Kwargs:
            x_solute_gb (ndarray): domain of the plot. Defaults to 0 to 0.5 (while the solute is a solute)
            filename (str): filename of the saved figure with extension. If no filename is provided, figure will not be saved.
            markers ([str]): list of matplotlib markers that can be overridden

        """
    energy = system.calculate_energy_for_xgb_d(overall_composition, temperature, grain_sizes, x_solute_gb)
    # Plot
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    for i, grain_size in enumerate(grain_sizes):
        plt.plot(x_solute_gb, energy[:][i], marker=markers[i], markersize=5, linestyle='None', label='d = {} nm'.format(str(grain_size)))
    ax.axhline(0, color='k')
    plt.legend(frameon=False, loc=0)
    plt.title(r'Grain Boundary Energy of {}-{}'.format(system.solvent, system.solute))
    plt.xlabel(system.solute + r' grain boundary mole fraction, $X^\mathrm{GB}_{\mathrm{'+system.solute+r'}}$', size=15)
    plt.ylabel(r'Normalized grain boundary energy, $\gamma/\gamma_0$', size=15)
    if filename:
        fig.savefig(filename)
        plt.close(fig)

def plot_grain_size_vs_temperature_for_x_overall(system, temperatures, overall_compositions,
                                                    filename=None, plot_inverse=True,
                                                    inverse_filename=None,
                                                    markers=['o', 'v', 'd', '^', '<', '>', 's', '*', 'x', '+', '1', '2']):
    """Plot a figure of stabilized grain size vs. temperature for different x_overall at fixed x_GB

    Args:
        temperatures (ndarray): temperature domain of the plot in Celsius
        overall_compositions (ndarray): array of compositions, this will be the number of lines plotted
    Kwargs:
        filename (str): filename of the saved figure with extension. If no filename is provided, figure will not be saved.
        plot_inverse (bool): whether to plot the inverse grain size plot
        inverse_filename (str): filename for an inverse plot. Will not work if plot_inverse is False. May change in future to effectively make plot_inverse true
        markers ([str]): list of matplotlib markers that can be overridden
    """

    grain_sizes = system.calculate_grain_size_for_temperature_x_overall(temperatures, overall_compositions)
    # Plot
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    for i, overall_composition in enumerate(overall_compositions):
        plt.plot(temperatures, grain_sizes[i][:], marker=markers[i], markersize=5, linestyle='--', label='$X_0 = $ {}'.format(overall_composition))
    ax.axhline(0, color='k')
    plt.legend(frameon=False, loc=0)
    plt.title(r'Stabilized Grain Sizes of {}-{}'.format(system.solvent, system.solute))
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

def plot_grain_size_vs_temperature_for_h_mix(system, x_overall, temperatures,
                                                h_mixes, filename=None,
                                                markers=['o', 'v', 'd', '^', '<', '>', 's', '*', 'x', '+', '1', '2']):
    """Plot a figure of stabilized grain size vs. temperature for different mixing enthalpies with fixed x_overall

    Args:
        x_overall (float): overall solute composition
        temperatures (ndarray): temperature domain of the plot in Celsius
        h_mixes (ndarray): array of compositions, this will be the number of lines plotted
    Kwargs:
        filename (str): filename of the saved figure with extension. If no filename is provided, figure will not be saved.
        markers ([str]): list of matplotlib markers that can be overridden
    """
    grain_sizes = system.calculate_grain_size_for_temperature_h_mix(x_overall, temperatures, h_mixes)
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
        (is_stable, overall_composition) = system.solute_can_stabilize(grain_size, max_solute_composition, temperature=temperature)
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
            #print("Ag-{}: stable concentration: {:03f}, shear modulus: {:d}, strengthening: {:d}".format(system.solute, composition, int(system.shear_modulus[system.solute]), int(system.calc_solid_solution_strengthening(composition))))
            print("{} {:0.5f}".format(system.solute, composition))
        else:
            plt.plot(system.h_elastic/1000, system.h_mix/1000, marker='o', color='k', markersize=((composition-composition_min)/(composition_max-composition_min)+1)*4, linestyle='')
        if label_points:
            plt.annotate('{}'.format(system.solute), xy=(system.h_elastic/1000, system.h_mix/1000), size=10, textcoords='data')
    plt.title(r'{} Solubility Map'.format(system.solvent))
    plt.xlabel(r'Elastic enthalpy, $\Delta H_{\mathrm{seg}}^\mathrm{{elastic}}$ $\mathrm{kJ/mol}$', size=15)
    plt.ylabel(r'Enthalpy of mixing, $\Delta H_{\mathrm{mix}}$ $\mathrm{kJ/mol}$', size=15)
    if interactive_plot:
        plt.show()
    if filename:
        plt.gcf().subplots_adjust(left=0.2)
        fig.savefig(filename)
        plt.close(fig)