# Sourced from https://github.com/sarahappleby/cgm/tree/master
# Edited by Matylda Rejus for SH 2025

# Plot our galaxy sample

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import numpy as np
import sys
import caesar
import os


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

def ssfr_b_redshift(z):
    return 1.9*np.log10(1+z) - 7.7

def belfiore_line(mstar, a=0.73, b=-7.7):
    # The definition of the SFMS from Belfiore+18 is:
    # log (SFR/Msun/yr) = 0.73 log (Mstar/Msun) - 7.33
    # With a scatter of sigma = 0.39 dex
    return mstar*a + b

def sfms_line(mstar, a=1., b=-10.8):
    return mstar*a + b

def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9.

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('plasma')
cmap = truncate_colormap(cmap, 0.1, 0.9)

cmap_r = plt.get_cmap('plasma_r')
cmap_r = truncate_colormap(cmap_r, 0.1, 0.9)

sf_cmap = plt.get_cmap('jet_r')
sf_cmap = truncate_colormap(sf_cmap, 0.1, 0.9)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    data_dir = f'/disk04/rad/sim/{model}/{wind}/'
    sim = caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5') 
    redshift = sim.simulation.redshift
    quench = quench_thresh(redshift)

    possible_snaps = ['151', '137', '125', '105', '078']
    snap_index = possible_snaps.index(snap)
    sf_height = [1.1, 1.2, 1.6, 2.1, 2.6]
    gv_height = [0.3, 0.3, 0.5, 0.7, 1.1]
    q_height = [-0.7, -0.7, -0.5, -0.3, 0.]
    q_height = [-3.05]*5
    ylims = [1.5, 1.75, 2., 2.5, 3.0]

    sample_dir = f'/disk04/mrejus/sh/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_sm = sf['mass'][:]
        gal_sfr = sf['sfr'][:]
        gal_ssfr = sf['ssfr'][:] +9.
        gal_fgas = np.log10(sf['fgas'][:] + 1e-3)
        gal_Lbaryon = sf['L_baryon'][:]
        #gal_nsats = sf['nsats'][:]
        gal_Tcgm = sf['Tcgm'][:]
        gal_fcold = sf['fcold'][:]

    print(f'Galaxies sampled: {len(gal_sm)}')
    z = np.array([0, 0, 1])
    Lbaryon_norm = gal_Lbaryon / np.linalg.norm(gal_Lbaryon, axis=1)[:, None]
    alpha = np.degrees(np.arccos([np.dot(z, i) for i in Lbaryon_norm]))

    # for face-on galaxies at 90 degrees and edge on galaxies at 0 degrees:
    #inclination = np.abs(alpha - 90)
    #for i in range(len(inclination)):
    #        if (inclination[i] > 90) & (inclination[i] < 180):
    #        inclination[i] = np.abs(inclination[i] - 180)

    # for face-on galaxies at 0 degrees and edge on galaxies at 90 degrees:
    inclination = alpha.copy()
    for i in range(len(inclination)):
        if (alpha[i] > 90) & (alpha[i] < 180):
            inclination[i] = 180 - alpha[i]

    delta_m = 0.25
    min_m = 10.
    nbins_m = 6
    mass_bins = np.arange(min_m, min_m+nbins_m*delta_m, delta_m)

    sm_line = np.arange(9.5, 12.5, 0.5)
    sf_line = sfms_line(sm_line,b=quench)
    q_line = sfms_line(sm_line, b=quench-1.)

    plot_dir = '/home/matylda/data/plots/'

    plt.plot(sm_line, sf_line, ls='--', lw=1.8, c='dimgray')
    plt.plot(sm_line, q_line, ls='--', lw=1.8, c='dimgray')
    plt.text(11.56, sf_height[snap_index], 'SF', fontsize=14)
    plt.text(11.55, gv_height[snap_index], 'GV', fontsize=14)
    plt.text(11.575, q_height[snap_index], 'Q', fontsize=14)
    for i in range(nbins_m + 1):
        plt.axvline(min_m+i*delta_m, ls=':', lw=1.5, c='darkgray')
    im = plt.scatter(gal_sm, np.log10(gal_sfr + 1e-3), c=gal_Tcgm, cmap=cmap, s=5, marker='o', vmin=5, vmax=7)
    plt.colorbar(im, label=r'${\rm log} (T_{\rm CGM} / {\rm K})$')
    plt.xlim(9.75,11.75)
    plt.ylim(-3.5, ylims[snap_index])
    plt.margins(y=0.05)
    plt.xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
    plt.ylabel(r'$\textrm{log} ({\rm SFR} / M_{\odot}{\rm yr}^{-1})$')
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Tcgm.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Tcgm.png', format='png',bbox_inches='tight')
    plt.close()

    plt.plot(sm_line, sf_line, ls='--', lw=1.3, c='dimgray')
    plt.plot(sm_line, q_line, ls='--', lw=1.3, c='dimgray')
    plt.text(11.56, sf_height[snap_index], 'SF')
    plt.text(11.55, gv_height[snap_index], 'GV')
    plt.text(11.575, q_height[snap_index], 'Q')
    for i in range(nbins_m + 1):
        plt.axvline(min_m+i*delta_m, ls=':', lw=1.5, c='darkgray')
    im = plt.scatter(gal_sm, np.log10(gal_sfr + 1e-3), c=gal_fcold, cmap=cmap_r, s=5, marker='o', vmin=-3, vmax=1)
    plt.colorbar(im, label=r'${\rm log} (M_{\rm cool} / M_{\rm hot})$')
    plt.xlim(9.75,11.75)
    plt.ylim(-3.5, ylims[snap_index])
    plt.xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
    plt.margins(y=0.05)
    plt.ylabel(r'$\textrm{log} ({\rm SFR} / M_{\odot}{\rm yr}^{-1})$')
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_fcold.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_fcold.png', format='png', bbox_inches='tight')
    plt.close()
