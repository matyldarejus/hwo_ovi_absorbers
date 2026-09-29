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
    norients = int(sys.argv[4]) # Needed to add this to make the number of lines of sight changeable

    vel_range = 600.
    
    """
    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5],
                      'snap_137': [3.5, 28.2, 10., 35.5, 8.0, 4.5],
                      'snap_125': [3.5, 31.6, 15.8, 39.8, 10., 5.6], 
                      'snap_105': [4.5, 25.1, 25.1, 34.5, 10., 7.1],}
    #chisq_lim_dict = {'snap_151': [3.5, 28.2, 15.8, 31.6, 5., 4.]} # for the extras sample
    """
    
    lines = ['OVI1031']
    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5]}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']
    
    snapfile = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    boxsize = float(s.boxsize.in_units_of('ckpc/h_0'))
    redshift = [s.redshift]

    delta_m = 0.5
    min_m = 10.
    nbins_m = 3
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    mass_bin_labels = []
    mass_plot_titles = []
    for i in range(nbins_m):
        mass_bin_labels.append(f'{mass_bins[i]}-{mass_bins[i+1]}')
        mass_plot_titles.append(f'{mass_bins[i]}'+ r'$ < \textrm{log} (M_* / M_{\odot}) < $' + f'{mass_bins[i+1]}')

    sim = caesar.load(f'/disk04/rad/sim/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    co = Cosmology(hubble_constant=sim.simulation.hubble_constant, omega_matter=sim.simulation.omega_matter, omega_lambda=sim.simulation.omega_lambda)
    hubble_parameter = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/Mpc')
    hubble_constant = co.hubble_parameter(0).in_units('km/s/Mpc')

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    ncells=16
    logN_min = 11.
    logN_max = 18.
    delta_logN = 0.5
    bins_logN = np.arange(logN_min, logN_max+delta_logN, delta_logN)
    bins_logN = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14., 15., 16., 17., 18.])
    plot_logN = get_bin_middle(bins_logN)
    delta_N = np.array([10**bins_logN[i+1] - 10**bins_logN[i] for i in range(len(plot_logN))])

    path_length_file = f'/disk04/mrejus/sh/path_lengths.h5'
    if not os.path.isfile(path_length_file):
        create_path_length_file(vel_range, lines, [redshift], path_length_file)
    path_lengths = read_h5_into_dict(path_length_file)

    plot_dir = f'~/data/plots'
    sample_dir = f'/disk04/mrejus/sh/samples/'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    #sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample_extras.h5'

    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
    
    nlos_all = len(gal_ids) * len(fr200) * norients
    nlos_low = len(gal_ids[mass < 10.5]) * len(fr200) * norients
    nlos_mid = len(gal_ids[(mass > 10.5) & (mass < 11.0)]) * len(fr200) * norients
    nlos_high = len(gal_ids[(mass > 11.0) & (mass < 11.5)]) * len(fr200) * norients

    dX_all = compute_dX(nlos_all, lines, path_lengths,
                    redshift=redshift, hubble_parameter=hubble_parameter,
                    hubble_constant=hubble_constant)[0]
    dX_mass = np.zeros(3)
    dX_mass[0] = compute_dX(nlos_low, lines, path_lengths,
                            redshift=redshift, hubble_parameter=hubble_parameter,
                            hubble_constant=hubble_constant)[0]
    dX_mass[1] = compute_dX(nlos_mid, lines, path_lengths,
                            redshift=redshift, hubble_parameter=hubble_parameter,
                            hubble_constant=hubble_constant)[0]
    dX_mass[2] = compute_dX(nlos_high, lines, path_lengths,
                            redshift=redshift, hubble_parameter=hubble_parameter,
                            hubble_constant=hubble_constant)[0]
    
    for l, line in enumerate(lines):

        results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_{line}.h5'
        cddf_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_{line}_cddf_mass.h5'

        #results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_fit_lines_{line}_extras.h5'
        #cddf_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_{line}_cddf_mass_extras.h5'

        plot_data = {}
        plot_data['plot_logN'] = plot_logN.copy()
        plot_data['bin_edges_logN'] = bins_logN.copy()

        all_N = []
        all_ew = []
        all_chisq = []
        all_ids = []
        all_los = []
            
        for j in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_N.extend(hf[f'log_N_{fr200[j]}r200'][:])
                all_ew.extend(hf[f'ew_{fr200[j]}r200'][:])
                all_chisq.extend(hf[f'chisq_{fr200[j]}r200'][:])
                all_ids.extend(hf[f'ids_{fr200[j]}r200'][:])
                all_los.extend(hf[f'LOS_pos_{fr200[j]}r200'][:])

        all_N = np.array(all_N)
        all_ew = np.array(all_ew)
        all_chisq = np.array(all_chisq)
        all_ids = np.array(all_ids)
        all_los = np.array(all_los)
    
        mask = (all_N > logN_min) * (all_chisq < chisq_lim[l]) * (all_ew >= 0.)
        all_N = all_N[mask]
        all_ew = all_ew[mask]
        all_los = all_los[mask]

        all_ids = all_ids[mask]
        #idx = np.where(gal_ids == some_id)
        idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
        all_mass = mass[idx]

        plot_data[f'cddf_all'] = np.zeros(len(plot_logN))
        
        
        for j in range(len(bins_logN) -1):
            N_mask = (all_N > bins_logN[j]) & (all_N < bins_logN[j+1])
            plot_data[f'cddf_all'][j] = len(all_N[N_mask])
        plot_data[f'cddf_all_poisson'] = np.sqrt(plot_data[f'cddf_all'])
        plot_data[f'cddf_all_poisson'] /= (plot_data[f'cddf_all'] * np.log(10.))
        plot_data[f'cddf_all'] /= (delta_N * dX_all)
        plot_data[f'cddf_all'] = np.log10(plot_data[f'cddf_all'])

        plot_data[f'cddf_all_cv_mean_{ncells}'], plot_data[f'cddf_all_cv_{ncells}'] = \
                get_cosmic_variance_cddf(all_N, all_los, boxsize, line, bins_logN, delta_N, path_lengths, ncells=ncells,
                                         redshift=redshift, hubble_parameter=hubble_parameter, hubble_constant=hubble_constant)

        for j in range(len(mass_bin_labels)):
            
            label = mass_bin_labels[j]
            plot_data[f'cddf_{label}'] = np.zeros(len(plot_logN))
            mass_mask = (all_mass > mass_bins[j]) & (all_mass < mass_bins[j+1])

            for k in range(len(bins_logN) -1):
                N_mask = (all_N > bins_logN[k]) & (all_N < bins_logN[k+1])
                plot_data[f'cddf_{label}'][k] = len(all_N[N_mask*mass_mask])

            plot_data[f'cddf_{label}_poisson'] = np.sqrt(plot_data[f'cddf_{label}'])
            plot_data[f'cddf_{label}_poisson'] /= (plot_data[f'cddf_{label}'] * np.log(10.))

            plot_data[f'cddf_{label}'] /= (delta_N * dX_mass[j])
            plot_data[f'cddf_{label}'] = np.log10(plot_data[f'cddf_{label}'])

            plot_data[f'cddf_{label}_cv_mean_{ncells}'], plot_data[f'cddf_{label}_cv_{ncells}'] = \
                    get_cosmic_variance_cddf(all_N[mass_mask], all_los[mass_mask], boxsize, line, bins_logN, delta_N, path_lengths, ncells=ncells,
                                            redshift=redshift, hubble_parameter=hubble_parameter, hubble_constant=hubble_constant)
    
        write_dict_to_h5(plot_data, cddf_file)
