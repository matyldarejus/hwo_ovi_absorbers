# Plot rho-T phase space diagrams for ALL gas particles in the simulation snapshots

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import h5py
import os
import sys
import pygad as pg
from pygadgetreader import readsnap

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    results_file = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}_phase_space.h5'

    sample_dir = f'/disk04/mrejus/sh/samples/'
    #snapfiles = [f'{sample_dir}{model}_{wind}_{snap}.hdf5', f'{sample_dir}{model}_{wind}_{snap}_star_forming.hdf5', 
                 #f'{sample_dir}{model}_{wind}_{snap}_green_valley.hdf5', f'{sample_dir}{model}_{wind}_{snap}_quenched.hdf5',]
    snapfiles = [f'{sample_dir}{model}_{wind}_{snap}.hdf5']
    #titles = ['All', 'Star forming', 'Green valley', 'Quenched']
    titles = ['All']

    Npoints = 20000
    minT = 3
    maxT = 8
    mindelta = -1
    maxdelta = 8
    width = 0.01
    height = 0.77
    vertical_position = 0.115
    horizontal_position = 0.92

    for i, snapfile in enumerate(snapfiles):
       
        results_file = snapfile.replace('.hdf5', '_phase_space.h5')
        plotfile = f'/home/matylda/data/plots/' + results_file.split('/')[-1].replace('.h5', '.png')

        if not os.path.isfile(results_file):
 
            s = pg.Snapshot(snapfile)
            redshift = s.redshift
            rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
            cosmic_rho = rho_crit * float(s.cosmology.Omega_b)
            h = s.cosmology.h_0
            zsolar = 0.0134 # in Zo

            over_bins = np.arange(mindelta, maxdelta+0.1, 0.1)
            temp_bins = np.arange(minT, maxT+0.1, 0.1) 

            gas_temp = np.log10(readsnap(snapfile, 'u', 'gas', suppress=1, units=1)) # in log K
            gas_rho = np.log10(readsnap(snapfile, 'rho', 'gas', suppress=1, units=1)) # in log g/cm^3
            gas_metal_frac = np.log10(readsnap(snapfile, 'Metallicity', 'gas', suppress=1, units=1)[:, 0] / zsolar)
            gas_overdensity = gas_rho - np.log10(cosmic_rho)
   
            hist = plt.hist2d(gas_overdensity, gas_temp, bins=[over_bins, temp_bins], norm=LogNorm(), density=True, cmap='Greys')

            plt.xlim(mindelta, maxdelta)
            plt.ylim(minT, maxT)
            plt.xlabel(r'${\rm log }\Delta$')
            plt.ylabel(r'${\rm log } (T / {\rm K})$')
            plt.title(titles[i])
            plt.colorbar(label='Frequency')
            plt.savefig(plotfile.replace('.png', '_hist.png'))
            plt.close()

            with h5py.File(results_file, 'a') as hf:
                hf.create_dataset('rho_delta_temp', data=np.array(np.rot90(hist[0])))
                hf.create_dataset('rho_delta_bins', data=np.array(over_bins))
                hf.create_dataset('temp_bins', data=np.array(temp_bins))

            over_bins = np.arange(mindelta, maxdelta+0.2, 0.2)
            temp_bins = np.arange(minT, maxT+0.2, 0.2)

            hist = plt.hist(gas_overdensity, bins=over_bins, density=True)
            with h5py.File(results_file, 'a') as hf:
                hf.create_dataset('rho_delta_hist', data=np.array(hist[0]))

            hist = plt.hist(gas_temp, bins=temp_bins, density=True)
            with h5py.File(results_file, 'a') as hf:
                hf.create_dataset('temp_hist', data=np.array(hist[0])) 

            mask = (gas_overdensity > mindelta) & (gas_overdensity < maxdelta) & (gas_temp > minT) & (gas_temp < maxT)

            gas_temp = gas_temp[mask]
            gas_overdensity = gas_overdensity[mask]
            gas_metal_frac = gas_metal_frac[mask]

            ids = np.random.choice(len(gas_temp), size=Npoints, replace=False)
            order = np.argsort(gas_overdensity)

            gas_temp = gas_temp[order][ids]
            gas_overdensity = gas_overdensity[order][ids]
            gas_metal_frac = gas_metal_frac[order][ids]

            im = plt.scatter(gas_overdensity, gas_temp, c=gas_metal_frac, cmap='magma', s=1, vmin=-2., vmax=0.5)
            plt.colorbar(im, label=r'${\rm log} (Z / Z_{\odot})$')
            plt.xlim(mindelta, maxdelta)
            plt.ylim(minT, maxT)
            plt.xlabel(r'${\rm log }\Delta$')
            plt.ylabel(r'${\rm log } (T / {\rm K})$')
            plt.title(titles[i])
            plt.savefig(plotfile.replace('.png', '_deltaTZ.png'))
            plt.close()
