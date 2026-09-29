import h5py
import numpy as np
import os
import sys
sys.path.insert(0, '/home/matylda/sh/make_spectra_sh/')
from utils import read_h5_into_dict

def equivalent_width(flux, pixel_size):
    return np.sum((np.ones(len(flux)) - flux) * pixel_size)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    norients = int(sys.argv[4]) 

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    # possible ions to choose:
    lines = ['OVI1031']
    orients = np.linspace(0, (360 - 360/norients), norients, dtype=int)
    
  #  for i in range(len(orients)):
  #      orient = str(orients[i])
  #      orients[i] = f'{orient}_deg'    
        
    vel_range = 600. #km/s
    vel_boxsize = 10000. # km/s
    bin_size = 6. # km/s 
    c = 2.98e8 # km/s
   
    spectra_dir = f'/disk04/mrejus/sh/normal/{model}_{wind}_{snap}_hm12/'
    sample_dir = f'/disk04/mrejus/sh/samples/' 
    
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    #sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample_extras.h5'
    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    for line in lines:

        results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_ew_{line}.h5'
        #results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_ew_{line}_extras.h5'

        if os.path.isfile(results_file):
            with h5py.File(results_file, 'r') as f:
                all_keys = list(f.keys())
        else:
            all_keys = []

        for i in range(len(fr200)):

            if not f'ew_wave_{fr200[i]}r200' in all_keys:

                all_ew = np.zeros((len(gal_ids), len(orients)))

                for j, gal in enumerate(gal_ids):
            
                    for o, orient in enumerate(orients):

                        spec_name = f'sample_galaxy_{gal_ids[j]}_{line}_{orient}_deg_{fr200[i]}r200'
                        spec = read_h5_into_dict(f'{spectra_dir}{spec_name}.h5')

                        if spec['gal_velocity_pos'] > vel_boxsize:
                            spec['gal_velocity_pos'] -= vel_boxsize
                        elif spec['gal_velocity_pos'] < 0:
                            spec['gal_velocity_pos'] += vel_boxsize
        
                        vel_mask = (spec['velocities'] < spec['gal_velocity_pos'] + vel_range) & (spec['velocities'] > spec['gal_velocity_pos'] - vel_range)
                        dv_right_edge = vel_boxsize - spec['gal_velocity_pos']
                        if dv_right_edge < vel_range:
                            vel_mask = vel_mask | (spec['velocities'] < vel_range - dv_right_edge)
                        dv_left_edge = spec['gal_velocity_pos'] - vel_range
                        if dv_left_edge < 0:
                            vel_mask = vel_mask | (spec['velocities'] > vel_boxsize - np.abs(dv_left_edge))

                        flux = spec['fluxes'][vel_mask]
                        pixel_size = spec['wavelengths'][1] - spec['wavelengths'][0]
                        all_ew[j][o] = equivalent_width(flux, pixel_size)

                with h5py.File(results_file, 'a') as f:
                    f.create_dataset(f'ew_wave_{fr200[i]}r200', data=np.array(all_ew))
            
