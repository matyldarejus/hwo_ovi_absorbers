import numpy as np
import sys
sys.path.insert(0, '/home/matylda/sh/make_spectra_sh/')
from physics import *


def variance_jk(samples, mean):
    n = len(samples)
    factor = (n-1.)/n
    x = np.zeros(len(mean))
    for i in range(len(x)):
        samples_use = samples[:, i]
        x[i] = np.nansum((np.subtract(samples_use[~np.isinf(samples_use)], mean[i]))**2, axis=0)
    x *= factor
    return x


def cell_indices_2d(los_array, boxsize, ncells=9):
    ncells_dimension = int(np.sqrt(ncells))
    gridsize = boxsize / ncells_dimension

    indices = {}
    count = 0

    for i in range(ncells_dimension):
        #print(los_array)
        x_mask = (los_array[:, 0] > gridsize*i) & (los_array[:, 0] < gridsize*(i+1))
        
        for j in range(ncells_dimension):
            y_mask = (los_array[:, 1] > gridsize*j) & (los_array[:, 1] < gridsize*(j+1))
            indices[f'indices_{count}'] = np.arange(len(los_array))[x_mask*y_mask]

            count += 1
   
    return indices


def get_cosmic_variance_ew(ew, los, boxsize, ncells=9):
    indices = cell_indices_2d(los, boxsize, ncells=ncells)
    ignore = np.arange(ncells) 
    medians = np.zeros(ncells)

    for i in range(ncells):
        using_mask = np.zeros(len(los), dtype=bool)
        for j in range(ncells):
            if j == ignore[i]: continue
            else:
                using_mask[indices[f'indices_{j}']] = True
        
        medians[i] = np.nanmedian(ew[using_mask])

    mean = np.nansum(medians) / ncells
    cosmic_std = np.sqrt(variance_jk(medians, mean))
    return mean, cosmic_std 


def get_cosmic_variance_cddf(N, los, boxsize, line, bins_logN, delta_N, path_lengths, ncells=9, redshift=0., hubble_parameter=68., hubble_constant=68.):
    indices = cell_indices_2d(los, boxsize, ncells=ncells)
    cddf = np.zeros((ncells, len(bins_logN)-1))
    ignore = np.arange(ncells)

    for i in range(ncells):
        using_mask = np.zeros(len(los), dtype=bool)
        for j in range(ncells):
            if j == ignore[i]: continue
            else:
                using_mask[indices[f'indices_{j}']] = True

        for j in range(len(bins_logN) -1):
            N_mask = (N > bins_logN[j]) & (N < bins_logN[j+1])
            cddf[i][j] = len(N[N_mask*using_mask])

    dX = np.array([compute_dX(i, [line], path_lengths, redshift=redshift, hubble_parameter=hubble_parameter, hubble_constant=hubble_constant)[0] for i in np.sum(cddf, axis=1)])

    dX = np.reshape(np.repeat(dX, len(bins_logN) -1), (ncells, len(bins_logN) -1))
    delta_N = np.reshape(np.tile(delta_N, ncells), (ncells, len(bins_logN) -1))
    cddf /= (delta_N * dX)
    cddf = np.log10(cddf)

    cddf_mean = np.zeros(cddf.shape[1])
    for i in range(len(cddf_mean)):
        cddf_use = cddf[:, i]
        cddf_mean[i] = np.nansum(cddf_use[~np.isinf(cddf_use)], axis=0) / ncells
    cosmic_std = np.sqrt(variance_jk(cddf, cddf_mean))

    return cddf_mean, cosmic_std

