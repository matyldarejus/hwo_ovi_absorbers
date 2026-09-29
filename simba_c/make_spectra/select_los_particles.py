# Script to identify LOS particles for a given selected galaxy, for LOS parallel to the z axis of the simulation
# Run using sub_los_particles.sh

# for each galaxy in the sample, identify particles that contribute towards the line of sight using the smoothing length of each particle
# save new dataset containing only these particles using Chris's approach

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
    x_dist = np.abs(los[0] - gas_pos[:, 0])
    y_dist = np.abs(los[1] - gas_pos[:, 1])
    hyp_sq = x_dist**2 + y_dist**2
    dist_mask = hyp_sq < hsml**2
    partids_los = np.arange(len(hsml))[dist_mask * wind_mask]
    return partids_los

if __name__ == '__main__':
    
    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    sample_gal = int(sys.argv[4]) # supply the gal id that we want from command line]
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

    #sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample_extras.h5'
    #particle_file = f'{sample_dir}{model}_{wind}_{snap}_particle_selection_extras.h5'

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


    """ The below code is meant for 8 lines of sight sampling only -- needed to be changed to get 
    better comparison for Habitable Worlds 
    for i in range(nbins_fr200):
        los = np.array([pos[:2].copy(), ]*8)
        rho = r200 * fr200[i]
        los[0][0] += rho
        los[1][0] += (rho / sqrt2); los[1][1] += (rho / sqrt2)
        los[2][1] += rho
        los[3][0] -= (rho / sqrt2); los[3][1] += (rho / sqrt2)
        los[4][0] -= rho
        los[5][0] -= (rho / sqrt2); los[5][1] -= (rho / sqrt2)
        los[6][1] -= rho
        los[7][0] += (rho / sqrt2); los[7][1] -= (rho / sqrt2)

        for l in los:
            partids_los = get_los_particles(l, gas_pos, hsml, wind_mask)
            partids = np.append(partids, partids_los)
            del partids_los

    """

    # Code for multiple lines of sight follows

    for i in range(nbins_fr200):
        rho = r200 * fr200[i]
        thetas = np.linspace(0, 2*np.pi, nlos, endpoint=False)
        for theta in thetas:
            los = pos[:2].copy()
            los[0] += rho * np.cos(theta)
            los[1] += rho * np.sin(theta)
            partids_los = get_los_particles(los, gas_pos, hsml, wind_mask)
            partids = np.append(partids, partids_los)
            

    partids = np.unique(np.sort(partids))
    with h5py.File(particle_file, 'a') as f:
        f.create_dataset(f'plist_{gal_id}', data=np.array(partids))
    del partids

