"""
Select the gas particles that contribute to the lines of sight around a galaxy.

For one galaxy in the sample, places nlos no. sightlines (parallel to the
simulation z-axis) at each of the five impact parameters (0.25-1.25 r200),
evenly spaced in azimuth. A gas particle is taken to contribute to a
sightline if its projected (x-y) distance from it is smaller than its
smoothing length, so its SPH kernel overlaps the sightline. Decoupled wind
particles (DelayTime != 0) are excluded.

The union of contributing particles over all sightlines is saved as
`plist_{gal_id}`, as indices into the snapshot's gas particle array, in
{model}_{wind}_{snap}_particle_selection.h5. Spectrum generation can then load
only these particles rather than the full snapshot, saving memory. Galaxies that already have
a `plist_` entry are skipped.

Run straight after selecting a galaxy sample (get_galaxy_sample.py) using 
sub_select_los_particles.sh. 
"""
from pygadgetreader import readsnap
import numpy as np
from numba import njit
import h5py
import caesar
import os
import gc
import sys

@njit
def get_los_particles(los, gas_pos, hsml, wind_mask):
    """
    Choose particles within the line of sight which overlap the LOS
    and are not wind particles
    """
    x_dist = np.abs(los[0] - gas_pos[:, 0]) # x offset of particle from LOS
    y_dist = np.abs(los[1] - gas_pos[:, 1]) # y offset
    hyp_sq = x_dist**2 + y_dist**2 
    dist_mask = hyp_sq < hsml**2 # offset needs to be smaller than the smoothing length
    partids_los = np.arange(len(hsml))[dist_mask * wind_mask]
    return partids_los

if __name__ == '__main__':
    
    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    sample_gal = int(sys.argv[4]) # allows to run via bash file 
    nlos = int(sys.argv[5])

    sqrt2 = np.sqrt(2.)
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    
    sample_dir = f'/disk04/mrejus/sh/samples/'
    data_dir = f'/disk04/rad/sim/m100n1024/simba-c/'
    snapfile = f'{data_dir}snap_{model}_{snap}.hdf5'
    
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    particle_file = f'{sample_dir}{model}_{wind}_{snap}_particle_selection.h5'

    sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
    h = sim.simulation.hubble_constant
    redshift = sim.simulation.redshift

    with h5py.File(sample_file, 'r') as f:
        gal_id = f['gal_ids'][:].astype('int')[sample_gal]
        pos = f['position'][:][sample_gal] * (1.+redshift) # already in kpc/h, factor of 1+z for comoving
        r200 = f['halo_r200'][:][sample_gal] * (1.+redshift) # already in kpc/h, factor of 1+z for comoving

    if os.path.isfile(particle_file):
        with h5py.File(particle_file, 'r') as hf:
            if f'plist_{gal_id}' in hf.keys():
                sys.exit()

    hsml = readsnap(snapfile, 'SmoothingLength', 'gas', suppress=1, units=1)  # in kpc/h, comoving
    gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) # in kpc/h, comoving
    gas_delaytime = readsnap(snapfile, 'DelayTime', 'gas', suppress=1)
    wind_mask = gas_delaytime == 0.

    partids = np.array([])

    for i in range(nbins_fr200): # for no. impact parameters
        rho = r200 * fr200[i] # projected distance of the sightline from the galaxy 
        thetas = np.linspace(0, 2*np.pi, nlos, endpoint=False) # get no. angles rel. to z-axis
        for theta in thetas:
            los = pos[:2].copy()
            los[0] += rho * np.cos(theta) # x
            los[1] += rho * np.sin(theta) # y
            partids_los = get_los_particles(los, gas_pos, hsml, wind_mask)
            partids = np.append(partids, partids_los)
            

    partids = np.unique(np.sort(partids))
    with h5py.File(particle_file, 'a') as f:
        f.create_dataset(f'plist_{gal_id}', data=np.array(partids))
    del partids

