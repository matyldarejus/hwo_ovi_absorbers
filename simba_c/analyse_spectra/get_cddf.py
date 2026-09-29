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

def bin_data(x, y, xbins):
    digitized = np.digitize(x, xbins)
    return np.array([y[digitized == i] for i in range(1, len(xbins))])

def quench_thresh(z): # in units of yr^-1 
    #print(z)
    return -1.8  + 0.3*z[0] -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask


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
    # Consider only snap = 151 and OVI1031 line

    lines = ['OVI1031']
    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5]}
    
    chisq_lim = chisq_lim_dict[f'snap_{snap}']

    snapfile = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    boxsize = float(s.boxsize.in_units_of('ckpc/h_0'))
    redshift = [s.redshift]
    quench = quench_thresh(redshift)

    sim = caesar.load(f'/disk04/rad/sim/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    co = Cosmology(hubble_constant=sim.simulation.hubble_constant, omega_matter=sim.simulation.omega_matter, omega_lambda=sim.simulation.omega_lambda)
    hubble_parameter = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/Mpc')
    hubble_constant = co.hubble_parameter(0).in_units('km/s/Mpc')

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    inner_outer = [[0.25, 0.5, 0.75], [1.0, 1.25]]
    labels = ['inner', 'outer']

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
        create_path_length_file(vel_range, lines, redshift, path_length_file)
    path_lengths = read_h5_into_dict(path_length_file)

    plot_dir = f'~/data/plots'
    sample_dir = f'/disk04/mrejus/sh/samples/'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    #sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample_extras.h5'

    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        ssfr = sf['ssfr'][:]
    
    gal_sf_mask, gal_gv_mask, gal_q_mask = ssfr_type_check(quench, ssfr) 
    nlos_all = len(gal_ids) * len(fr200) * norients
    nlos_sf = len(gal_ids[gal_sf_mask]) * len(fr200) * norients
    nlos_gv = len(gal_ids[gal_gv_mask]) * len(fr200) * norients
    nlos_q = len(gal_ids[gal_q_mask]) * len(fr200) * norients

    dX_all = compute_dX(nlos_all, lines, path_lengths,
                    redshift=redshift, hubble_parameter=hubble_parameter,
                    hubble_constant=hubble_constant)[0]
    dX_sf = compute_dX(nlos_sf, lines, path_lengths,
                    redshift=redshift, hubble_parameter=hubble_parameter,
                    hubble_constant=hubble_constant)[0]
    dX_gv = compute_dX(nlos_gv, lines, path_lengths,
                    redshift=redshift, hubble_parameter=hubble_parameter,
                    hubble_constant=hubble_constant)[0]
    dX_q = compute_dX(nlos_q, lines, path_lengths,
                    redshift=redshift, hubble_parameter=hubble_parameter,
                    hubble_constant=hubble_constant)[0]

    #print("dX_all, dX_sf, dX_gv, dX_q =", dX_all, dX_sf, dX_gv, dX_q)

    for l, line in enumerate(lines):

        results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_{line}.h5'
        cddf_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_{line}_cddf_chisqion.h5'

        #print(results_file)

        #results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_fit_lines_{line}_extras.h5'
        #cddf_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_{line}_cddf_chisqion_extras.h5'

        plot_data = {}
        plot_data['plot_logN'] = plot_logN.copy()
        plot_data['bin_edges_logN'] = bins_logN.copy()

        all_N = []
        all_b = []
        all_l = []
        all_ew = []
        all_chisq = []
        all_ids = []
        all_los = []
            
        for j in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_N.extend(hf[f'log_N_{fr200[j]}r200'][:])
                all_b.extend(hf[f'b_{fr200[j]}r200'][:])
                all_l.extend(hf[f'l_{fr200[j]}r200'][:])
                all_ew.extend(hf[f'ew_{fr200[j]}r200'][:])
                all_chisq.extend(hf[f'chisq_{fr200[j]}r200'][:])
                all_ids.extend(hf[f'ids_{fr200[j]}r200'][:])
                all_los.extend(hf[f'LOS_pos_{fr200[j]}r200'][:])

        all_N = np.array(all_N)
        all_b = np.array(all_b)
        all_l = np.array(all_l)
        all_ew = np.array(all_ew)
        all_chisq = np.array(all_chisq)
        all_ids = np.array(all_ids)
        all_los = np.array(all_los)

        #print(all_N)
        #print(all_b)
        #print(all_l)
        #print(all_ew)

        mask = (all_N > logN_min) * (all_chisq < chisq_lim[l]) * (all_ew >= 0.)
        all_N = all_N[mask]
        all_b = all_b[mask]
        all_l = all_l[mask]
        all_ew = all_ew[mask]
        all_los = all_los[mask]

        all_ids = all_ids[mask]
        #idx = np.where(gal_id == some_id)
        #idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
        idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
        all_ssfr = ssfr[idx]

        sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

        plot_data[f'cddf_all'] = np.zeros(len(plot_logN))
        plot_data[f'cddf_sf'] = np.zeros(len(plot_logN))
        plot_data[f'cddf_gv'] = np.zeros(len(plot_logN))
        plot_data[f'cddf_q'] = np.zeros(len(plot_logN))

        for j in range(len(bins_logN) -1):
            N_mask = (all_N > bins_logN[j]) & (all_N < bins_logN[j+1])
            plot_data[f'cddf_all'][j] = len(all_N[N_mask])
            plot_data[f'cddf_sf'][j] = len(all_N[N_mask*sf_mask])
            plot_data[f'cddf_gv'][j] = len(all_N[N_mask*gv_mask])
            plot_data[f'cddf_q'][j] = len(all_N[N_mask*q_mask])

        plot_data[f'cddf_all_poisson'] = np.sqrt(plot_data[f'cddf_all'])
        plot_data[f'cddf_all_poisson'] /= (plot_data[f'cddf_all'] * np.log(10.))

        plot_data[f'cddf_sf_poisson'] = np.sqrt(plot_data[f'cddf_sf'])
        plot_data[f'cddf_sf_poisson'] /= (plot_data[f'cddf_sf'] * np.log(10.))

        plot_data[f'cddf_gv_poisson'] = np.sqrt(plot_data[f'cddf_gv'])
        plot_data[f'cddf_gv_poisson'] /= (plot_data[f'cddf_gv'] * np.log(10.))

        plot_data[f'cddf_q_poisson'] = np.sqrt(plot_data[f'cddf_q'])
        plot_data[f'cddf_q_poisson'] /= (plot_data[f'cddf_q'] * np.log(10.))

        plot_data[f'cddf_all'] /= (delta_N * dX_all)
        plot_data[f'cddf_sf'] /= (delta_N * dX_sf)
        plot_data[f'cddf_gv'] /= (delta_N * dX_gv)
        plot_data[f'cddf_q'] /= (delta_N * dX_q)
        plot_data[f'cddf_all'] = np.log10(plot_data[f'cddf_all'])
        plot_data[f'cddf_sf'] = np.log10(plot_data[f'cddf_sf'])
        plot_data[f'cddf_gv'] = np.log10(plot_data[f'cddf_gv'])
        plot_data[f'cddf_q'] = np.log10(plot_data[f'cddf_q'])

        print(plot_data[f'cddf_all'])
        
        plot_data[f'cddf_all_cv_mean_{ncells}'], plot_data[f'cddf_all_cv_{ncells}'] = \
                get_cosmic_variance_cddf(all_N, all_los, boxsize, line, bins_logN, delta_N, path_lengths, ncells=ncells, 
                                         redshift=redshift, hubble_parameter=hubble_parameter, hubble_constant=hubble_constant)

        plot_data[f'cddf_sf_cv_mean_{ncells}'], plot_data[f'cddf_sf_cv_{ncells}'] = \
                get_cosmic_variance_cddf(all_N[sf_mask], all_los[sf_mask], boxsize, line, bins_logN, delta_N, path_lengths, ncells=ncells, 
                                         redshift=redshift, hubble_parameter=hubble_parameter, hubble_constant=hubble_constant)

        plot_data[f'cddf_gv_cv_mean_{ncells}'], plot_data[f'cddf_gv_cv_{ncells}'] = \
                get_cosmic_variance_cddf(all_N[gv_mask], all_los[gv_mask], boxsize, line, bins_logN, delta_N, path_lengths, ncells=ncells,
                                         redshift=redshift, hubble_parameter=hubble_parameter, hubble_constant=hubble_constant)

        plot_data[f'cddf_q_cv_mean_{ncells}'], plot_data[f'cddf_q_cv_{ncells}'] = \
                get_cosmic_variance_cddf(all_N[q_mask] , all_los[q_mask], boxsize, line, bins_logN, delta_N, path_lengths, ncells=ncells,
                                         redshift=redshift, hubble_parameter=hubble_parameter, hubble_constant=hubble_constant)


        #binned_N = bin_data(all_N, all_N, bins_logN)
        #plot_data['cddf_all_std'] = np.array([np.nanstd(i) for i in binned_N])

        
        for i in range(len(inner_outer)):

            nlos_all = len(gal_ids) * len(inner_outer[i]) * norients
            nlos_sf = len(gal_ids[gal_sf_mask]) * len(inner_outer[i]) * norients
            nlos_gv = len(gal_ids[gal_gv_mask]) * len(inner_outer[i]) * norients
            nlos_q = len(gal_ids[gal_q_mask]) * len(inner_outer[i]) * norients

            dX_all = compute_dX(nlos_all, [line], path_lengths,
                                redshift=redshift, hubble_parameter=hubble_parameter,
                                hubble_constant=hubble_constant)[0]
            dX_sf = compute_dX(nlos_sf, [line], path_lengths,
                                redshift=redshift, hubble_parameter=hubble_parameter,
                                hubble_constant=hubble_constant)[0]
            dX_gv = compute_dX(nlos_gv, [line], path_lengths,
                                redshift=redshift, hubble_parameter=hubble_parameter,
                                hubble_constant=hubble_constant)[0]
            dX_q = compute_dX(nlos_q, [line], path_lengths,
                                redshift=redshift, hubble_parameter=hubble_parameter,
                                hubble_constant=hubble_constant)[0]

            all_N = []
            all_b = []
            all_l = []
            all_ew = []
            all_chisq = []
            all_ids = []

            for j in range(len(inner_outer[i])):
                
                with h5py.File(results_file, 'r') as hf:
                    all_N.extend(hf[f'log_N_{inner_outer[i][j]}r200'][:])
                    all_b.extend(hf[f'b_{inner_outer[i][j]}r200'][:])
                    all_l.extend(hf[f'l_{inner_outer[i][j]}r200'][:])
                    all_ew.extend(hf[f'ew_{inner_outer[i][j]}r200'][:])
                    all_chisq.extend(hf[f'chisq_{inner_outer[i][j]}r200'][:])
                    all_ids.extend(hf[f'ids_{inner_outer[i][j]}r200'][:])

            all_N = np.array(all_N)
            all_b = np.array(all_b)
            all_l = np.array(all_l)
            all_ew = np.array(all_ew)
            all_chisq = np.array(all_chisq)
            all_ids = np.array(all_ids)

            mask = (all_N > logN_min) * (all_chisq < chisq_lim[l])
            all_N = all_N[mask]
            all_b = all_b[mask]
            all_l = all_l[mask]
            all_ew = all_ew[mask]
                
            all_ids = all_ids[mask]
            idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
            all_ssfr = ssfr[idx]

            sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

            plot_data[f'cddf_all_{labels[i]}'] = np.zeros(len(plot_logN))
            plot_data[f'cddf_sf_{labels[i]}'] = np.zeros(len(plot_logN))
            plot_data[f'cddf_gv_{labels[i]}'] = np.zeros(len(plot_logN))
            plot_data[f'cddf_q_{labels[i]}'] = np.zeros(len(plot_logN))

            for j in range(len(bins_logN)-1):
                N_mask = (all_N > bins_logN[j]) & (all_N < bins_logN[j+1])
                plot_data[f'cddf_all_{labels[i]}'][j] = len(all_N[N_mask])
                plot_data[f'cddf_sf_{labels[i]}'][j] = len(all_N[N_mask*sf_mask])
                plot_data[f'cddf_gv_{labels[i]}'][j] = len(all_N[N_mask*gv_mask])
                plot_data[f'cddf_q_{labels[i]}'][j] = len(all_N[N_mask*q_mask])

            plot_data[f'cddf_all_{labels[i]}'] /= (delta_N * dX_all)
            plot_data[f'cddf_sf_{labels[i]}'] /= (delta_N * dX_sf)
            plot_data[f'cddf_gv_{labels[i]}'] /= (delta_N * dX_gv)
            plot_data[f'cddf_q_{labels[i]}'] /= (delta_N * dX_q)

            plot_data[f'cddf_all_{labels[i]}'] = np.log10(plot_data[f'cddf_all_{labels[i]}'])
            plot_data[f'cddf_sf_{labels[i]}'] = np.log10(plot_data[f'cddf_sf_{labels[i]}'])
            plot_data[f'cddf_gv_{labels[i]}'] = np.log10(plot_data[f'cddf_gv_{labels[i]}'])
            plot_data[f'cddf_q_{labels[i]}'] = np.log10(plot_data[f'cddf_q_{labels[i]}'])
        

        write_dict_to_h5(plot_data, cddf_file)
