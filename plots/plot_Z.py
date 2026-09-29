import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib.colors as colors
import numpy as np
import h5py
import pygad as pg
import sys
 
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
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.frameon'] = True
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.dpi'] = 130
 
def quench_thresh(z):  # in units of yr^-1
    return -1.8 + 0.3*z - 9.
 
def ssfr_type_check(ssfr_thresh, ssfr):
    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh - 1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask
 
 
if __name__ == '__main__':
 
    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
 
    line = "OVI1031"
    plot_line = r'${\rm OVI}\ 1031$'
    
    chisq_lim_dict = {'snap_151': 4.5,
                      'snap_137': 4.5,
                      'snap_125': 5.6,
                      'snap_105': 7.1}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']
    N_min = 13.2
    zsolar = 5.79e-3  # Solar oxygen abundance
 
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200 + 1) * delta_fr200, delta_fr200)
 
    snapfile = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)
    quench = quench_thresh(redshift)
 
    plot_dir = '/home/matylda/data/plots/'
    sample_dir = '/disk04/mrejus/sh/samples/'
    results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_{line}.h5'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
 
    # Load in data
    with h5py.File(sample_file, 'r') as sf:
        gal_ids  = sf['gal_ids'][:]
        gal_mass = sf['mass'][:]
        gal_ssfr = sf['ssfr'][:]
 
        # ISM metallicities
        Z_mass_weighted = sf['Z_mass_weighted'][:]  # mass-weighted
        Z_sfr_weighted  = sf['Z_sfr_weighted'][:]   # SFR-weighted
        Z_stellar       = sf['Z_stellar'][:]         # stellar metallicity
 
        # CGM metallicities
        Z_cgm_mw        = sf['Z_cgm_mw'][:]         # CGM mass-weighted
        Z_cgm_tw        = sf['Z_cgm_tw'][:]          # CGM temp-weighted
 
    # Convert to Z/Zsolar
    Z_mass_weighted_sol = np.log10(Z_mass_weighted) - np.log10(zsolar)
    Z_sfr_weighted_sol  = np.log10(Z_sfr_weighted)  - np.log10(zsolar)
    Z_cgm_mw_sol        = np.log10(Z_cgm_mw)        - np.log10(zsolar)
    Z_cgm_tw_sol        = np.log10(Z_cgm_tw)        - np.log10(zsolar)
 
    all_Z_absorbers    = []
    all_N_absorbers    = []
    all_chisq_absorbers = []
    all_ids_absorbers  = []
 
    with h5py.File(results_file, 'r') as hf:
        for i, r in enumerate(fr200):
            try:
                all_Z_absorbers.extend(hf[f'log_Z_{r}r200'][:])
                all_N_absorbers.extend(hf[f'log_N_{r}r200'][:])
                all_chisq_absorbers.extend(hf[f'chisq_{r}r200'][:])
                all_ids_absorbers.extend(hf[f'ids_{r}r200'][:])
            except KeyError:
                print(f"Warning: Data not found for {r}r200")
                continue
 
    all_Z_absorbers     = np.array(all_Z_absorbers)
    all_N_absorbers     = np.array(all_N_absorbers)
    all_chisq_absorbers = np.array(all_chisq_absorbers)
    all_ids_absorbers   = np.array(all_ids_absorbers)
 
    # Apply quality cuts
    mask = (all_N_absorbers > N_min) & (all_chisq_absorbers < chisq_lim)
    all_Z_absorbers   = all_Z_absorbers[mask]
    all_ids_absorbers = all_ids_absorbers[mask]
 
    # Match absorbers to galaxy masses
    idx = np.array([np.where(gal_ids == gid)[0][0] for gid in all_ids_absorbers
                    if gid in gal_ids])
    absorber_masses = gal_mass[idx]
    absorber_Z      = all_Z_absorbers - np.log10(zsolar)
 
    nmin = 15
 
    # Plot 1: Absorbers vs CGM metallicities
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
 
    # Absorbers
    ax.scatter(absorber_masses, absorber_Z, c='#5289C7', s=3, alpha=0.5,
               label='Absorbers', zorder=1)
    bin_cent, ymean, ysiglo, ysighi, ndata = pm.runningmedian(
        absorber_masses, absorber_Z, stat='median')
    ax.plot(bin_cent[ndata > nmin], ymean[ndata > nmin],
            c='#5289C7', lw=2.5, ls='-', label='Absorbers (median)')
 
    # CGM mass-weighted
    ax.scatter(gal_mass, Z_cgm_mw_sol, c='#90C987', s=15, alpha=0.7,
               marker='s', label='CGM mass-weighted', zorder=2)
    bin_cent, ymean, ysiglo, ysighi, ndata = pm.runningmedian(
        gal_mass, Z_cgm_mw_sol, stat='median')
    ax.plot(bin_cent[ndata > nmin], ymean[ndata > nmin],
            c='#90C987', lw=2.5, ls='--', label='CGM mass-weighted (median)')
 
    # CGM temp-weighted
    ax.scatter(gal_mass, Z_cgm_tw_sol, c='#F4A460', s=15, alpha=0.7,
               marker='D', label='CGM temp-weighted', zorder=2)
    bin_cent, ymean, ysiglo, ysighi, ndata = pm.runningmedian(
        gal_mass, Z_cgm_tw_sol, stat='median')
    ax.plot(bin_cent[ndata > nmin], ymean[ndata > nmin],
            c='#F4A460', lw=2.5, ls='--', label='CGM temp-weighted (median)')
 
    ax.axhline(0, ls=':', c='k', lw=1.5, label=r'$Z_{\odot}$')
    ax.set_xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
    ax.set_ylabel(r'$\log\ (Z / Z_{\odot})$')
    ax.set_xlim(9.75, 11.75)
    ax.set_ylim(-2, 1)
    ax.legend(loc='best', fontsize=12, markerscale=0.7, handlelength=1.5)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_absorbers_vs_cgm_metallicity.png',
                format='png', dpi=400)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_absorbers_vs_cgm_metallicity.pdf',
                format='pdf')
    plt.close()
    print(f"Saved: absorbers_vs_cgm_metallicity")
 
    # Plot 2: Absorbers vs ISM/galaxy metallicities
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
 
    # Absorbers
    ax.scatter(absorber_masses, absorber_Z, c='#5289C7', s=3, alpha=0.5,
               label='Absorbers', zorder=1)
    bin_cent, ymean, ysiglo, ysighi, ndata = pm.runningmedian(
        absorber_masses, absorber_Z, stat='median')
    ax.plot(bin_cent[ndata > nmin], ymean[ndata > nmin],
            c='#5289C7', lw=2.5, ls='-', label='Absorbers (median)')
 
    # SFR-weighted
    ax.scatter(gal_mass, Z_sfr_weighted_sol, c='#E26F72', s=15, alpha=0.7,
               marker='^', label='ISM SFR-weighted', zorder=3)
    bin_cent, ymean, ysiglo, ysighi, ndata = pm.runningmedian(
        gal_mass, Z_sfr_weighted_sol, stat='median')
    ax.plot(bin_cent[ndata > nmin], ymean[ndata > nmin],
            c='#E26F72', lw=2.5, ls='-.', label='ISM SFR-weighted (median)')
 
    # Mass-weighted galaxy
    ax.scatter(gal_mass, Z_mass_weighted_sol, c='#9B59B6', s=15, alpha=0.7,
               marker='o', label='ISM mass-weighted', zorder=3)
    bin_cent, ymean, ysiglo, ysighi, ndata = pm.runningmedian(
        gal_mass, Z_mass_weighted_sol, stat='median')
    ax.plot(bin_cent[ndata > nmin], ymean[ndata > nmin],
            c='#9B59B6', lw=2.5, ls='-.', label='ISM mass-weighted (median)')
 
 
    ax.axhline(0, ls=':', c='k', lw=1.5, label=r'$Z_{\odot}$')
    ax.set_xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
    ax.set_ylabel(r'$\log\ (Z / Z_{\odot})$')
    ax.set_xlim(9.75, 11.75)
    ax.set_ylim(-2, 1)
    ax.legend(loc='best', fontsize=12, markerscale=0.7, handlelength=1.5)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_absorbers_vs_ism_metallicity.png',
                format='png', dpi=400)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_absorbers_vs_ism_metallicity.pdf',
                format='pdf')
    plt.close()