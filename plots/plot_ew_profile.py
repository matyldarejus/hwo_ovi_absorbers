# Sourced from https://github.com/sarahappleby/cgm/tree/master
# Edited by Matylda Rejus for SH 2025
# Plots the equivalent width profiles for different ssfr types of galaxies

import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import caesar

sys.path.insert(0, '/home/matylda/sh/make_spectra_sh/')
from utils import read_h5_into_dict, write_dict_to_h5

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 27
plt.rcParams['axes.titlesize'] = 27
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['legend.fontsize'] = 23
plt.rcParams['legend.frameon'] = True
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.dpi'] = 130

cb_blue = '#5289C7'
cb_green = '#90C987'
cb_red = '#E26F72'

def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask



if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    norients = int(sys.argv[4])
    verbose = True if sys.argv[5] == 'True' else False
    
    data_dir = f'/disk04/rad/sim/{model}/{wind}/'
    sim = caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift
    quench = quench_thresh(redshift)

    """
    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    """
    lines = ["OVI1031"]
    plot_lines = [r'${\rm OVI}\ 1031$']
    plot_quantities = ['sf_med', 'sf_per25', 'sf_per75', 'gv_med', 'gv_per25', 'gv_per75', 'q_med', 'q_per25', 'q_per75',]

    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    chisq_lim = 2.5

    delta_m = 0.5
    min_m = 10.
    nbins_m = 3
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    
    mass_bin_labels = [] 
    mass_plot_titles = []
    for i in range(nbins_m):
        mass_bin_labels.append(f'{mass_bins[i]}-{mass_bins[i+1]}')
        mass_plot_titles.append(f'{mass_bins[i]}'+ r'$ < \textrm{log} (M_* / M_{\odot}) < $' + f'{mass_bins[i+1]}')

    results_dir = f'/disk04/mrejus/sh/normal/results/'
    plot_dir = f'/home/matylda/data/plots/'
    sample_dir = f'/disk04/mrejus/sh/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        gal_sm = sf['mass'][:]
        gal_ssfr = sf['ssfr'][:]

    mass_long = np.repeat(gal_sm, norients)
    
    if verbose:
        print(mass_long)
    ssfr_long = np.repeat(gal_ssfr, norients)
    sf_mask, gv_mask, q_mask = ssfr_type_check(quench, ssfr_long)

    fig, ax = plt.subplots(len(lines), nbins_m, figsize=(14, 13), constrained_layout=False, sharey='row', sharex='col')

    if len(lines) == 1:
        ax = np.array([ax])

    for l, line in enumerate(lines):

        profile_file = f'{results_dir}{model}_{wind}_{snap}_{line}_median_ew_profile.h5'

        if os.path.isfile(profile_file):
            plot_data = read_h5_into_dict(profile_file)
        else:

            ew_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_ew_{line}.h5')

            plot_data = {}
            plot_data['fr200'] = fr200.copy()
            for bl in mass_bin_labels:
                for pq in plot_quantities:
                    plot_data[f'{bl}_{pq}'] = np.zeros(len(fr200))

            for i, bin_label in enumerate(mass_bin_labels):

                mass_mask = (mass_long > mass_bins[i]) & (mass_long < mass_bins[i+1])

                for j in range(len(fr200)):
                    ew = ew_dict[f'ew_wave_{fr200[j]}r200'].flatten()

                    plot_data[f'{bin_label}_sf_med'][j] = np.nanmedian(np.log10(ew[sf_mask*mass_mask]))
                    plot_data[f'{bin_label}_sf_per25'][j] = np.nanpercentile(np.log10(ew[sf_mask*mass_mask]), 25)
                    plot_data[f'{bin_label}_sf_per75'][j] = np.nanpercentile(np.log10(ew[sf_mask*mass_mask]), 75)

                    plot_data[f'{bin_label}_gv_med'][j] = np.nanmedian(np.log10(ew[gv_mask*mass_mask]))
                    plot_data[f'{bin_label}_gv_per25'][j] = np.nanpercentile(np.log10(ew[gv_mask*mass_mask]), 25)
                    plot_data[f'{bin_label}_gv_per75'][j] = np.nanpercentile(np.log10(ew[gv_mask*mass_mask]), 75)

                    plot_data[f'{bin_label}_q_med'][j] = np.nanmedian(np.log10(ew[q_mask*mass_mask]))
                    plot_data[f'{bin_label}_q_per25'][j] = np.nanpercentile(np.log10(ew[q_mask*mass_mask]), 25)
                    plot_data[f'{bin_label}_q_per75'][j] = np.nanpercentile(np.log10(ew[q_mask*mass_mask]), 75)

            write_dict_to_h5(plot_data, profile_file)


        for b, bin_label in enumerate(mass_bin_labels):

            ax[l][b].plot(plot_data['fr200'], plot_data[f'{bin_label}_sf_med'], ls='-', c=cb_blue, label='Star-forming', lw=2.5)
            ax[l][b].fill_between(plot_data['fr200'], plot_data[f'{bin_label}_sf_per75'], plot_data[f'{bin_label}_sf_per25'], alpha=0.3, color=cb_blue)

            ax[l][b].plot(plot_data['fr200'], plot_data[f'{bin_label}_gv_med'], ls='--', c=cb_green, label='Green valley', lw=2.5)
            ax[l][b].fill_between(plot_data['fr200'], plot_data[f'{bin_label}_gv_per75'], plot_data[f'{bin_label}_gv_per25'], alpha=0.3, color=cb_green)

            ax[l][b].plot(plot_data['fr200'], plot_data[f'{bin_label}_q_med'], ls=:-', c=cb_red, label='Quenched', lw=2.5)
            ax[l][b].fill_between(plot_data['fr200'], plot_data[f'{bin_label}_q_per75'], plot_data[f'{bin_label}_q_per25'], alpha=0.3, color=cb_red)

            ax[l][b].set_ylim(-3., 0.)

            if b == 0:
                ax[l][b].set_ylabel(r'${\rm log (EW}/\AA)$')
                ax[l][b].set_title(mass_plot_titles[b])
                ax[l][b].legend(loc=1)

            if l == 0:
                ax[l][b].set_title(mass_plot_titles[b])

            if l == len(lines) -1 and b == 1:
                ax[l][b].set_xlabel(r'$r_\perp / r_{200}$')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_ew_profile.png', dpi=400)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_ew_profile.pdf')
    plt.show()
    plt.clf()
