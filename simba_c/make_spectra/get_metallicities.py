import numpy as np
import h5py
import sys
import caesar
 
 
if __name__ == '__main__':
 
    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
 
    sample_dir = f'/disk04/mrejus/sh/samples/'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
 
    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]
 
    data_dir = f'/disk04/rad/sim/m100n1024/simba-c/'
    sim = caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
 

    Z_mass_weighted = np.array([sim.galaxies[i].metallicities['mass_weighted']
                                 for i in range(len(sim.galaxies))])[gal_ids]
 
    Z_sfr_weighted  = np.array([sim.galaxies[i].metallicities['sfr_weighted']
                                 for i in range(len(sim.galaxies))])[gal_ids]
 
    Z_stellar       = np.array([sim.galaxies[i].metallicities['stellar']
                                 for i in range(len(sim.galaxies))])[gal_ids]
 

    Z_cgm_mass_weighted = np.array([sim.galaxies[i].halo.metallicities['mass_weighted_cgm']
                                     for i in range(len(sim.galaxies))])[gal_ids]
 
    Z_cgm_temp_weighted = np.array([sim.galaxies[i].halo.metallicities['temp_weighted_cgm']
                                     for i in range(len(sim.galaxies))])[gal_ids]
 

    with h5py.File(sample_file, 'a') as sf:
        # ISM / galaxy metallicities
        sf.create_dataset('Z_mass_weighted', data=Z_mass_weighted)
        sf.create_dataset('Z_sfr_weighted',  data=Z_sfr_weighted)
        sf.create_dataset('Z_stellar',       data=Z_stellar)
        # CGM metallicities
        sf.create_dataset('Z_cgm_mw',        data=Z_cgm_mass_weighted)
        sf.create_dataset('Z_cgm_tw',        data=Z_cgm_temp_weighted)
 
    print(f'Metallicities written to {sample_file}')