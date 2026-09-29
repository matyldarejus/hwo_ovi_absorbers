import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import h5py
import yt
import caesar
import sys

if __name__ == '__main__':
    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    
    data_dir = f'/disk04/rad/sim/{model}/{wind}/'
    sim = caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')

    results_file = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}_sm_sfr.h5'  
    results_file_ssfr = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}_sm_ssfr.h5'

    mass_min = 9.75
    mass_max = 11.75
    ssfr_max = 0.
    ssfr_min = -3.8

    nbins = 50
    delta_m = (mass_max - mass_min) / nbins
    delta_ssfr = (ssfr_max - ssfr_min) / nbins
    mass_bins = np.arange(mass_min, mass_max+delta_m, delta_m)
    ssfr_bins = np.arange(ssfr_min, ssfr_max+delta_ssfr, delta_ssfr)

    gal_cent = np.array([i.central for i in sim.galaxies])
    gal_sm = yt.YTArray([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))], 'Msun')
    gal_sfr = yt.YTArray([sim.galaxies[i].sfr.in_units('Msun/Gyr') for i in range(len(sim.galaxies))], 'Msun/yr')
    gal_ssfr = gal_sfr / gal_sm 
    gal_ssfr = np.log10(gal_ssfr.value + 10**ssfr_min)
    gal_sm = np.log10(gal_sm)

    hist_ssfr = plt.hist2d(gal_sm[gal_cent], gal_ssfr[gal_cent], bins=[mass_bins, ssfr_bins], norm=LogNorm(), density=True, cmap='Greys')

    with h5py.File(results_file_ssfr, 'a') as hf:
        hf.create_dataset('sm_ssfr', data=np.array(np.rot90(hist_ssfr[0])))
        hf.create_dataset('mass_bins', data=np.array(mass_bins))
        hf.create_dataset('ssfr_bins', data=np.array(ssfr_bins))    