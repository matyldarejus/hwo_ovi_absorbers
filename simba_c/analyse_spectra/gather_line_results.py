import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy import interpolate
import pygad as pg
import os
import sys
sys.path.insert(0, '/home/matylda/sh/simba_c/make_spectra_sh/')
from utils import *
from physics import *

def get_interp_conditions(wavelengths, lines, quantity):
    wave_boxsize = wavelengths[-1] - wavelengths[0]
    for i in range(len(lines)):
        if lines[i] < np.min(wavelengths):
            lines[i] += wave_boxsize
        elif lines[i] > np.max(wavelengths):
            lines[i] -= wave_boxsize

    model = interpolate.interp1d(wavelengths, quantity)
    return model(lines)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    line = "OVI1031"
    fr200 = float(sys.argv[4])
    norients = int(sys.argv[5])
    
    vel_range = 600. #km/s
    #orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg'] 
    orients = np.linspace(0, (360 - 360/norients), norients, dtype=int)
    

    sample_dir = f'/disk04/mrejus/sh/samples/'
    #spectra_dir = f'/disk04/mrejus/sh/normal/{model}_{wind}_{snap}/'
    spectra_dir = f'/disk04/mrejus/sh/normal/{model}_{wind}_{snap}_hm12/'
    #results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'
    #results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_fit_lines_{line}_extras.h5'
    results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_{line}.h5'

    s = pg.Snapshot(f'{sample_dir}{model}_{wind}_{snap}.hdf5')
    #s = pg.Snapshot(f'{sample_dir}{model}_{wind}_{snap}_extras.hdf5')
    redshift = s.redshift

    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    #sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample_extras.h5'

    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    all_rho = []
    all_T = []
    all_Z = []
    all_Nspec = []
    all_vpec = []
    all_los = []
    all_orient = []

    all_pos_dv = []

    all_N = []
    all_b = []
    all_l = []
    all_ew = []
    all_chisq = []
    all_ids = []
    all_ts = []

    n_found = 0

    gal_todo = []
    for i in range(len(gal_ids)):
        for o, orient in enumerate(orients):
            spec_name = f'sample_galaxy_{gal_ids[i]}_{line}_{orient}_deg_{fr200}r200'
            spectrum = read_h5_into_dict(f'{spectra_dir}{spec_name}.h5')

            #print(spec_name)

            
            # Check if lines are present
            if 'line_list' not in spectrum or 'N' not in spectrum['line_list']:
                print(f"[!] No line_list in {spec_name}")
                gal_todo.append(spec_name)
                continue

            n_lines = len(spectrum['line_list']['N'])
        
            if n_lines == 0:
                #print(f"[ ] No detected lines in {spec_name}")
                continue
    
            n_found += n_lines
            #print(f"[+] {spec_name}: {n_lines} lines")


            if not 'line_list' in spectrum.keys():
                spectrum['line_list'] = {}
                spectrum['line_list']['N'] = []
            
        
            if len(spectrum['line_list']['N']) > 0.:
             
                wave_boxsize = spectrum['wavelengths'][-1] - spectrum['wavelengths'][0]
                for j in range(len(spectrum['line_list']['l'])):
                    if spectrum['line_list']['l'][j] < np.min(spectrum['wavelengths']):
                        spectrum['line_list']['l'][j] += wave_boxsize
                    elif spectrum['line_list']['l'][j] > np.max(spectrum['wavelengths']):
                        spectrum['line_list']['l'][j] -= wave_boxsize

                spectrum['line_list']['v'] = np.array(wave_to_vel(spectrum['line_list']['l'], spectrum['lambda_rest'], redshift))                
                line_mask = (spectrum['line_list']['v'] > spectrum['gal_velocity_pos'] - vel_range) & (spectrum['line_list']['v'] < spectrum['gal_velocity_pos'] + vel_range)

                for j in range(len(spectrum['line_list']['l'][line_mask])):
                    index = np.argmin(np.abs(spectrum['wavelengths'] - spectrum['line_list']['l'][line_mask][j]))
                    all_rho.append(np.log10(spectrum['phys_density'][index]))
                    all_T.append(np.log10(spectrum['temperature'][index]))
                    all_Z.append(spectrum['metallicity'][index])
                    all_Nspec.append(np.log10(spectrum['col_density'][index]))
                    all_vpec.append(spectrum['vpec'][index])

                    all_los.extend(spectrum['LOS_pos'][:2])
                    all_orient.append(orient)

                all_pos_dv.extend(np.array(wave_to_vel(spectrum['line_list']['l'][line_mask], spectrum['lambda_rest'], redshift)) - spectrum['gal_velocity_pos'])

                all_chisq.extend(spectrum['line_list']['Chisq'][line_mask])
                all_N.extend(spectrum['line_list']['N'][line_mask])
                #print(f"N: {spectrum['line_list']['N'][line_mask]}")
                all_b.extend(spectrum['line_list']['b'][line_mask])
                all_l.extend(spectrum['line_list']['l'][line_mask])
                all_ew.extend(spectrum['line_list']['EW'][line_mask])
                all_ids.extend([gal_ids[i]] * len(spectrum['line_list']['N'][line_mask]))
                all_ts.extend(spectrum['line_list']['t'][line_mask])

    all_los = np.reshape(all_los, (int(len(all_los)*0.5), 2))

    # Optionally: Wipe the old file

    #if os.path.exists(results_file):
    #    os.remove(results_file)
    #print(len(gal_todo))   
    with h5py.File(results_file, 'a') as hf:
        if not f'log_rho_{fr200}r200' in hf.keys():
            hf.create_dataset(f'log_rho_{fr200}r200', data=np.array(all_rho))
            print(f'Added log_rho_{fr200}r200')
        if not f'log_T_{fr200}r200' in hf.keys():
            hf.create_dataset(f'log_T_{fr200}r200', data=np.array(all_T))
        if not f'log_Z_{fr200}r200' in hf.keys():
            hf.create_dataset(f'log_Z_{fr200}r200', data=np.array(all_Z))
        if not f'log_Nspec_{fr200}r200' in hf.keys():
            hf.create_dataset(f'log_Nspec_{fr200}r200', data=np.array(all_Nspec))
        if not f'vpec_{fr200}r200' in hf.keys():
            hf.create_dataset(f'vpec_{fr200}r200', data=np.array(all_vpec))
        if not f'LOS_pos_{fr200}r200' in hf.keys():
            hf.create_dataset(f'LOS_pos_{fr200}r200', data=np.array(all_los))
        if not f'orient_{fr200}r200' in hf.keys():
            hf.create_dataset(f'orient_{fr200}r200', data=np.array(all_orient))
        if not f'pos_dv_{fr200}r200' in hf.keys():
            hf.create_dataset(f'pos_dv_{fr200}r200', data=np.array(all_pos_dv))
        if not f'log_N_{fr200}r200' in hf.keys():
            print(f'Added log_N_{fr200}r200')
            hf.create_dataset(f'log_N_{fr200}r200', data=np.array(all_N))
        if not f'b_{fr200}r200' in hf.keys():
            hf.create_dataset(f'b_{fr200}r200', data=np.array(all_b))
        if not f'l_{fr200}r200' in hf.keys():
            hf.create_dataset(f'l_{fr200}r200', data=np.array(all_l))
        if not f'ew_{fr200}r200' in hf.keys():
            hf.create_dataset(f'ew_{fr200}r200', data=np.array(all_ew))
        if not f'chisq_{fr200}r200' in hf.keys():
            hf.create_dataset(f'chisq_{fr200}r200', data=np.array(all_chisq))
        if not f'ids_{fr200}r200' in hf.keys():
            hf.create_dataset(f'ids_{fr200}r200', data=np.array(all_ids))
        if not f't_{fr200}r200' in hf.keys():
            hf.create_dataset(f't_{fr200}r200', data=np.array(all_ts))