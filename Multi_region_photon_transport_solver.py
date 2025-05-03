import numpy as np
import pandas as pd
import sympy as sym
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import math
import statistics
######################################################################################################
def solve_bte(num_regions, region_widths, region_sources, region_sigma_t, region_sigma_s, num_angles):
    # Ensure high precision using numpy.float64
    dtype = np.float64

    # Total number of cells
    total_cells = sum(region_widths)
    dx = dtype(1.0)  # Cell width (spatial step size)

    # Initialize arrays for flux, source, and cross-sections
    flux = np.zeros(total_cells, dtype=dtype)
    source = np.zeros(total_cells, dtype=dtype)
    sigma_t = np.zeros(total_cells, dtype=dtype)
    sigma_s = np.zeros(total_cells, dtype=dtype)

    # Populate the source and cross-section arrays based on regions
    cell_index = 0
    for region in range(num_regions):
        for _ in range(region_widths[region]):
            source[cell_index] = dtype(region_sources[region])
            sigma_t[cell_index] = dtype(region_sigma_t[region])
            sigma_s[cell_index] = dtype(region_sigma_s[region])
            cell_index += 1

    # Use Gauss-Legendre quadrature for angular integration
    mu_values, weights = np.polynomial.legendre.leggauss(num_angles)
    angular_flux = np.zeros((total_cells, len(mu_values)), dtype=dtype)

    # Iteration parameters
    max_iterations = 1000
    tolerance = dtype(1e-10)
    old_flux = np.zeros_like(flux)
######################################################################################################
    # Source iteration loop
    for iteration in range(max_iterations):
        old_flux[:] = flux[:]
        
        # Angular sweep
        for n, mu in enumerate(mu_values):
            if mu > 0:  # Forward sweep
                angular_flux[0, n] = source[0] / (sigma_t[0] + mu/dx)
                for i in range(1, total_cells):
                    # Include streaming, collision, and source terms
                    angular_flux[i, n] = (
                        mu/dx * angular_flux[i-1, n] + 
                        source[i] + sigma_s[i] * flux[i]/2
                    ) / (sigma_t[i] + mu/dx)
            else:  # Backward sweep
                angular_flux[-1, n] = source[-1] / (sigma_t[-1] - mu/dx)
                for i in range(total_cells-2, -1, -1):
                    angular_flux[i, n] = (
                        -mu/dx * angular_flux[i+1, n] + 
                        source[i] + sigma_s[i] * flux[i]/2
                    ) / (sigma_t[i] - mu/dx)

        # Update scalar flux
        flux = np.sum(angular_flux * weights[:, np.newaxis].T, axis=1)

        # Check convergence
        if np.allclose(flux, old_flux, rtol=tolerance, atol=tolerance):
            print(f"Converged in {iteration + 1} iterations.")
            break
    else:
        print("Warning: Iterations did not converge within the maximum limit.")

    return flux
######################################################################################################
if __name__ == "__main__":
    # Define input parameters for a multi-region problem
    num_regions = 5
    region_widths = [20, 20, 20, 20, 20]  # Number of cells in each region
    region_sources = [0.1, 1, 5, 1, 0.1]  # Source strength in each region
    region_sigma_t = [0.01, 0.2, 5, 0.2, 0.01]  # Total macroscopic cross-sections for each region
    region_sigma_s = [0.001, 0.15, 4.89, 0.15, 0.001]  # Scattering macroscopic cross-sections for each region
    num_angles = 16  # Number of angles

    # Solve the BTE
    flux = solve_bte(num_regions, region_widths, region_sources, region_sigma_t, region_sigma_s, num_angles)
    # Print the photon flux distribution
    print("Cell index | Photon flux")
    print("-----------------------------")
    spatial_cells = np.arange(sum(region_widths))
    for cell, value in zip(spatial_cells, flux):
        print(f"{cell:10} | {value:.12f}")
    print("-------------------------------------------------------")
    print("--------------------Simulation ends--------------------")
    print("-------------------------------------------------------")
######################################################################################################
plt.figure(figsize=(10,10), dpi=100)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.linewidth'] = 1.5
plt.rc("font", size=25, family="Arial", weight='bold')    
plt.scatter(spatial_cells, flux, color='Red',linewidth=1.5)
plt.minorticks_on()
plt.tick_params(axis='both', which='major', length=15, width=2,labelsize=22)
plt.tick_params(axis='both', which='minor', length=7.5, width=1.5,labelsize=22)
plt.xticks(np.arange(0, 100.01, step=5))
plt.yticks(np.arange(0, 100.01, step=10))
plt.xlim(0.0, 100.01)
plt.ylim(1.0, 100.01)
plt.xlabel(r"\bf Cell number", fontsize=27)
plt.ylabel(r"\bf Photon flux [cm$^{-2}$s$^{-1}]$", fontsize=27)
plt.savefig("Photon_flux.pdf", bbox_inches='tight')
plt.show()       
