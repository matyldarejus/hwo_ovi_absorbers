import numpy as np
import h5py
import pygad as pg
import caesar
from yt.utilities.cosmology import Cosmology
import os
import sys

sys.path.insert(0, '/home/matylda/sh/make_spectra_sh/')
from utils import *
from physics import create_path_length_file, compute_dX
from cosmic_variance import get_cosmic_variance_cddf


def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    norients = int(sys.argv[4])

    vel_range = 600.
    lines = ['OVI1031']

    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5]}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']

    snapfile = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    boxsize = float(s.boxsize.in_units_of('ckpc/h_0'))
    redshift = [s.redshift]

    sim = caesar.load(f'/disk04/rad/sim/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    co = Cosmology(hubble_constant=sim.simulation.hubble_constant,
                   omega_matter=sim.simulation.omega_matter,
                   omega_lambda=sim.simulation.omega_lambda)
    hubble_parameter = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/Mpc')
    hubble_constant = co.hubble_parameter(0).in_units('km/s/Mpc')

    # Define azimuthal bins (angle between LOS and angular momentum)
    az_bins = np.array([0, 30, 90])
    az_labels = ['major', 'minor']

    ncells = 16
    logN_min = 11.
    bins_logN = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14., 15., 16., 17., 18.])
    plot_logN = get_bin_middle(bins_logN)
    delta_N = np.array([10**bins_logN[i+1] - 10**bins_logN[i] for i in range(len(plot_logN))])

    # Path length file
    path_length_file = f'/disk04/mrejus/sh/path_lengths.h5'
    if not os.path.isfile(path_length_file):
        create_path_length_file(vel_range, lines, [redshift], path_length_file)
    path_lengths = read_h5_into_dict(path_length_file)

    # Load galaxy sample file
    sample_dir = f'/disk04/mrejus/sh/samples/'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'

    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        gal_pos = sf['position'][:]
        gal_L = sf['L_baryon'][:]  # angular momentum vectors

    # Normalise angular momentum vectors
    L_norm = np.linalg.norm(gal_L, axis=1)
    gal_L_unit = gal_L / L_norm[:, None]

    # Compute path length normalization
    nlos_all = len(gal_ids) * norients
    dX_all = compute_dX(nlos_all, lines, path_lengths,
                        redshift=redshift, hubble_parameter=hubble_parameter,
                        hubble_constant=hubble_constant)[0]

    for l, line in enumerate(lines):

        results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_{line}.h5'
        cddf_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_{line}_cddf_azimuth.h5'

        plot_data = {}
        plot_data['plot_logN'] = plot_logN.copy()
        plot_data['bin_edges_logN'] = bins_logN.copy()

        all_N, all_ew, all_chisq, all_ids, all_los = [], [], [], [], []

        fr200_values = [0.25, 0.5, 0.75, 1.0, 1.25]

        for fr in fr200_values:
            with h5py.File(results_file, 'r') as hf:
                all_N.extend(hf[f'log_N_{fr}r200'][:])
                all_ew.extend(hf[f'ew_{fr}r200'][:])
                all_chisq.extend(hf[f'chisq_{fr}r200'][:])
                all_ids.extend(hf[f'ids_{fr}r200'][:])
                all_los.extend(hf[f'LOS_pos_{fr}r200'][:])

        all_N = np.array(all_N)
        all_ew = np.array(all_ew)
        all_chisq = np.array(all_chisq)
        all_ids = np.array(all_ids)
        all_los = np.array(all_los)

        mask = (all_N > logN_min) & (all_chisq < chisq_lim[l]) & (all_ew >= 0.)
        all_N, all_los, all_ids = all_N[mask], all_los[mask], all_ids[mask]

        all_los = np.column_stack((all_los, np.zeros(len(all_los))))  # add z=0 for 2D positions

        # Compute angles between LOS vector and L_baryon
        gal_index = np.array([np.where(gal_ids == g)[0][0] for g in all_ids])
        los_vecs = all_los - gal_pos[gal_index]
        los_vecs /= np.linalg.norm(los_vecs, axis=1)[:, None]

        cos_theta = np.sum(los_vecs * gal_L_unit[gal_index], axis=1)
        theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

        # All-sightlines CDDF
        plot_data['cddf_all'] = np.zeros(len(plot_logN))
        for j in range(len(bins_logN) - 1):
            N_mask = (all_N > bins_logN[j]) & (all_N < bins_logN[j + 1])
            plot_data['cddf_all'][j] = len(all_N[N_mask])
        plot_data['cddf_all_poisson'] = np.sqrt(plot_data['cddf_all']) / (plot_data['cddf_all'] * np.log(10.))
        plot_data['cddf_all'] /= (delta_N * dX_all)
        plot_data['cddf_all'] = np.log10(plot_data['cddf_all'])

        plot_data[f'cddf_all_cv_mean_{ncells}'], plot_data[f'cddf_all_cv_{ncells}'] = \
            get_cosmic_variance_cddf(all_N, all_los, boxsize, line, bins_logN, delta_N,
                                     path_lengths, ncells=ncells, redshift=redshift,
                                     hubble_parameter=hubble_parameter, hubble_constant=hubble_constant)

        # === AZIMUTHAL bins ===
        for j, label in enumerate(az_labels):
            plot_data[f'cddf_{label}'] = np.zeros(len(plot_logN))
            az_mask = (theta >= az_bins[j]) & (theta < az_bins[j+1])

            for k in range(len(bins_logN) - 1):
                N_mask = (all_N > bins_logN[k]) & (all_N < bins_logN[k + 1])
                plot_data[f'cddf_{label}'][k] = len(all_N[N_mask & az_mask])

            # Poisson + normalization
            plot_data[f'cddf_{label}_poisson'] = np.sqrt(plot_data[f'cddf_{label}'])
            plot_data[f'cddf_{label}_poisson'] /= (plot_data[f'cddf_{label}'] * np.log(10.))
            plot_data[f'cddf_{label}'] /= (delta_N * dX_all)
            plot_data[f'cddf_{label}'] = np.log10(plot_data[f'cddf_{label}'])

            plot_data[f'cddf_{label}_cv_mean_{ncells}'], plot_data[f'cddf_{label}_cv_{ncells}'] = \
                get_cosmic_variance_cddf(all_N[az_mask], all_los[az_mask], boxsize, line, bins_logN,
                                         delta_N, path_lengths, ncells=ncells,
                                         redshift=redshift, hubble_parameter=hubble_parameter,
                                         hubble_constant=hubble_constant)

        write_dict_to_h5(plot_data, cddf_file)
