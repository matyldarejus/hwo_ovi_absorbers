import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib.colors as colors
import numpy as np
import h5py
import pygad as pg
import sys
sys.path.insert(0, '/home/matylda/sh/make_spectra_sh/')
from utils import *
from physics import *
sys.path.append('/home/matylda/sh/tools/')
import plotmedian as pm
 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.frameon'] = True
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.dpi'] = 130
 
def quench_thresh(z):
    return -1.8 + 0.3*z - 9.
 
def ssfr_type_check(ssfr_thresh, ssfr):
    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh - 1)
    q_mask  = ssfr == -14.0
    return sf_mask, gv_mask, q_mask
 
def plot_median_band(ax, x, y, color, ls, lw, nmin, label=None, alpha=0.15):
    """Plot running median with 16-84th percentile shading."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < nmin:
        return
    # Use fewer bins when working within a narrow mass range
    nbins = max(5, len(x) // 20)
    bc, ymed, ylo, yhi, ndata = pm.runningmedian(x, y, nbins=nbins, stat='median')
    mask = ndata > nmin
    if np.sum(mask) == 0:
        return
    ax.plot(bc[mask], ymed[mask], c=color, ls=ls, lw=lw, label=label)
    ax.fill_between(bc[mask], ylo[mask], yhi[mask], color=color, alpha=alpha)
 
 
if __name__ == '__main__':
 
    model = sys.argv[1]
    wind  = sys.argv[2]
    snap  = sys.argv[3]
 
    zsolar    = 5.79e-3
    chisq_lim = 4.5
    N_min     = 13.2
    nmin      = 10
    nmin_panel = 3
 
    delta_fr200 = 0.25
    min_fr200   = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200 + 1) * delta_fr200, delta_fr200)
 
    snapfile = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}.hdf5'
    s        = pg.Snapshot(snapfile)
    redshift = s.redshift
    quench   = quench_thresh(redshift)
 
    plot_dir   = '/home/matylda/data/plots/'
    sample_dir = '/disk04/mrejus/sh/samples/'
    results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_OVI1031.h5'
    sample_file  = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
 
    # Load galaxy data
    with h5py.File(sample_file, 'r') as sf:
        gal_ids        = sf['gal_ids'][:]
        mass           = sf['mass'][:]
        ssfr           = sf['ssfr'][:]
        Z_sfr_weighted = sf['Z_sfr_weighted'][:]
        Z_mass_weighted = sf['Z_mass_weighted'][:]
        Z_cgm_mw       = sf['Z_cgm_mw'][:]
 
    #Z_ism_sol = np.log10(Z_sfr_weighted) - np.log10(zsolar)
    Z_ism_sol = np.log10(Z_mass_weighted) - np.log10(zsolar)
    Z_cgm_sol = np.log10(Z_cgm_mw)       - np.log10(zsolar)
 
    # Load absorber data
    all_Z_abs   = []
    all_N_abs   = []
    all_chisq   = []
    all_ids_abs = []
 
    with h5py.File(results_file, 'r') as hf:
        for r in fr200:
            try:
                all_Z_abs.extend(hf[f'log_Z_{r}r200'][:])
                all_N_abs.extend(hf[f'log_N_{r}r200'][:])
                all_chisq.extend(hf[f'chisq_{r}r200'][:])
                all_ids_abs.extend(hf[f'ids_{r}r200'][:])
            except KeyError:
                continue
 
    all_Z_abs   = np.array(all_Z_abs)
    all_N_abs   = np.array(all_N_abs)
    all_chisq   = np.array(all_chisq)
    all_ids_abs = np.array(all_ids_abs)
 
    mask        = (all_N_abs > N_min) & (all_chisq < chisq_lim)
    all_Z_abs   = all_Z_abs[mask] - np.log10(zsolar)
    all_ids_abs = all_ids_abs[mask]
 
    valid       = np.isin(all_ids_abs, gal_ids)
    all_Z_abs   = all_Z_abs[valid]
    all_ids_abs = all_ids_abs[valid]
    idx         = np.array([np.where(gal_ids == gid)[0][0] for gid in all_ids_abs])
    abs_mass    = mass[idx]
    abs_ssfr    = ssfr[idx]
 
    # Line styles: ISM=solid, CGM=dashed, Absorbers=dotted
    ism_ls    = '-'
    cgm_ls    = '--'
    abs_ls    = ':'
    lw        = 2.5
    ism_color = '#E26F72'  # red
    cgm_color = '#5289C7'  # blue
    abs_color = '#90C987'  # green
 
    legend_entries = [
        Line2D([0,1],[0,1], color=ism_color, ls=ism_ls, lw=lw, label='ISM (mass-weighted)'),
        Line2D([0,1],[0,1], color=cgm_color, ls=cgm_ls, lw=lw, label='CGM (mass-weighted)'),
        Line2D([0,1],[0,1], color=abs_color, ls=abs_ls, lw=lw, label=r'OVI Absorbers'),
    ]
 
    # FIGURE 1: Mass bins
    delta_m   = 0.5
    min_m     = 10.
    nbins_m   = 3
    mass_bins = np.arange(min_m, min_m + (nbins_m + 1) * delta_m, delta_m)
    mass_titles = [f'${mass_bins[i]:.1f}'
                   + r'< \log (M_{\star} / M_{\odot}) <'
                   + f'{mass_bins[i+1]:.1f}$'
                   for i in range(nbins_m)]
 
    fig, axes = plt.subplots(1, nbins_m, figsize=(15, 5), sharey=True, sharex=False)
 
    for k, ax in enumerate(axes):
        mlo, mhi = mass_bins[k], mass_bins[k+1]
        gal_mask = (mass >= mlo) & (mass < mhi)
        ab_mask  = (abs_mass >= mlo) & (abs_mass < mhi)

        plot_median_band(ax, mass[gal_mask],    Z_ism_sol[gal_mask], ism_color, ism_ls, lw, nmin_panel)
        plot_median_band(ax, mass[gal_mask],    Z_cgm_sol[gal_mask], cgm_color, cgm_ls, lw, nmin_panel)
        plot_median_band(ax, abs_mass[ab_mask], all_Z_abs[ab_mask],  abs_color, abs_ls, lw, nmin_panel)

        ax.axhline(0, ls=':', c='k', lw=1, alpha=0.5)
        ax.set_title(mass_titles[k])
        ax.set_xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
        ax.set_xlim(mlo, mhi)
        ax.set_ylim(-2, 1)
 
    axes[0].set_ylabel(r'$\log\ (Z / Z_{\odot})$')
    axes[0].legend(handles=legend_entries, loc='lower right', fontsize=16)
 
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Z_medians_mass_bins.png', format='png', dpi=400)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Z_medians_mass_bins.pdf', format='pdf')
    plt.close()
    print('Saved: Z_medians_mass_bins')
 
    # FIGURE 2: SSFR Bins
    sf_mask_gal, gv_mask_gal, q_mask_gal = ssfr_type_check(quench, ssfr)
    sf_mask_abs, gv_mask_abs, q_mask_abs = ssfr_type_check(quench, abs_ssfr)
 
    ssfr_titles = ['Star-forming', 'Green valley', 'Quenched']
    gal_masks   = [sf_mask_gal, gv_mask_gal, q_mask_gal]
    abs_masks   = [sf_mask_abs, gv_mask_abs, q_mask_abs]
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
 
    for k, ax in enumerate(axes):
        gmask = gal_masks[k]
        amask = abs_masks[k]
 
        plot_median_band(ax, mass[gmask],     Z_ism_sol[gmask], ism_color, ism_ls, lw, nmin_panel)
        plot_median_band(ax, mass[gmask],     Z_cgm_sol[gmask], cgm_color, cgm_ls, lw, nmin_panel)
        plot_median_band(ax, abs_mass[amask], all_Z_abs[amask], abs_color, abs_ls, lw, nmin_panel)
 
        ax.axhline(0, ls=':', c='k', lw=1, alpha=0.5)
        ax.set_title(ssfr_titles[k])
        ax.set_xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
        ax.set_xlim(9.75, 11.75)
        ax.set_ylim(-2, 1)
 
    axes[0].set_ylabel(r'$\log\ (Z / Z_{\odot})$')
    axes[0].legend(handles=legend_entries, loc='lower right', fontsize=16)
 
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Z_medians_ssfr_bins.png', format='png', dpi=400)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Z_medians_ssfr_bins.pdf', format='pdf')
    plt.close()
    print('Saved: Z_medians_ssfr_bins')