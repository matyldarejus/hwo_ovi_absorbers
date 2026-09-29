# Saving out the halo temperature properties of the sample galaxies

import numpy as np
import h5py
import sys
import caesar
from pygadgetreader import readsnap

def ism_phase_line(nH):
    # ISM particles have:
    # log T  = 4 + 1/3 log nH  (+ 1 dex)
    return 5. + 0.33*nH

def get_ism_mask(temp, nH, ism_density):
    nH_mask = nH > ism_density
    ism_line = ism_phase_line(np.log10(nH))
    temp_mask = (np.log10(temp) - ism_line < 0.)
    return temp_mask * nH_mask


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    photo_temp = 10.**4.5 # in K
    ism_density = 0.13 # hydrogen number density, cm**-3

    sample_dir = f'/disk04/mrejus/sh/samples/'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5' # Use the galaxy samples generated and append
    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]
 
    data_dir = f'/disk04/rad/sim/m100n1024/simba-c/'
    sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
    h = sim.simulation.hubble_constant

    Tcgm = np.log10([i.halo.temperatures['mass_weighted_cgm'].in_units('K') for i in sim.galaxies])[gal_ids]

    snapfile = f'{data_dir}snap_{model}_{snap}.hdf5' 
    gas_mass = readsnap(snapfile, 'mass', 'gas', suppress=1, units=1) / h # in Mo
    gas_nh = readsnap(snapfile, 'nh', 'gas', suppress=1, units=1) # in g/cm^3
    gas_delaytime = readsnap(snapfile, 'DelayTime', 'gas', suppress=1)
    gas_temp = readsnap(snapfile, 'u', 'gas', suppress=1, units=1) # in K

    cold_mass = np.zeros(len(gal_ids))
    hot_mass = np.zeros(len(gal_ids))

    for i, gal_id in enumerate(gal_ids):
        glist = sim.galaxies[gal_id].halo.glist 

        ism_gas_mask = get_ism_mask(gas_temp[glist], gas_nh[glist], ism_density)
        cgm_gas_mask = np.invert(ism_gas_mask)
        wind_mask = gas_delaytime[glist] > 0.

        cool_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] < photo_temp)
        hot_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > photo_temp)
        
        cold_mass[i] = np.sum(gas_mass[glist][cool_gas_mask])
        hot_mass[i] = np.sum(gas_mass[glist][hot_gas_mask])

    fcold = np.log10(cold_mass / hot_mass + 1e-3)

    with h5py.File(sample_file, 'a') as sf:
        sf.create_dataset('Tcgm', data=np.array(Tcgm))
        sf.create_dataset('mcold', data=np.array(cold_mass))
        sf.create_dataset('mhot', data=np.array(hot_mass))
        sf.create_dataset('fcold', data=np.array(fcold))

    print(f'Update the attributes to include the temperature')
