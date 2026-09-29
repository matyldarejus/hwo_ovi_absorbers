# Script for creating subsets of particles from gizmo sims
# Based on scripts by Sydney Lower and Chris Lovell
# https://gist.github.com/christopherlovell/5a504c2c9d26efb6e073324d80c755a6 


import h5py
import caesar
import numpy as np
import sys

def make_new_dataset(snapfile, output_file, plist, verbose):
    ignore_fields = []
    with h5py.File(snapfile, 'r') as in_file:

        for ptype in ['PartType0']:

            pidx = int(ptype[8:]) # get particle type index

            for k in in_file[ptype]: # loop through fields

                if k in ignore_fields:
                    if verbose > 1: print(k, ' skipped...')
                    continue

                if verbose > 1: print(ptype, k)

                # load a given field (the bottleneck)
                temp_dset = in_file[ptype][k][:]
                if verbose > 1: print(temp_dset.shape)


                with h5py.File(output_file, 'a') as out_file:

                    if k in out_file[ptype]:
                        if verbose > 1: 
                            print("dataset already exists. replacing...")
                        del out_file[ptype][k]

                    out_file[ptype].create_dataset(k, data = temp_dset[plist])

                    temp = out_file['Header'].attrs['NumPart_ThisFile']
                    temp[pidx] = len(plist)
                    out_file['Header'].attrs['NumPart_ThisFile'] = temp

                    temp = out_file['Header'].attrs['NumPart_Total']
                    temp[pidx] = len(plist)
                    out_file['Header'].attrs['NumPart_Total'] = temp


def prepare_out_file(snapfile, output_file, numpart):
    with h5py.File(snapfile, 'r') as in_file:
        header = in_file['Header']

        with h5py.File(output_file, 'a') as out_file:
            if 'Header' not in out_file:
                out_file.copy(header, 'Header')
                out_file['Header'].attrs['NumPart_ThisFile'] = numpart
                out_file['Header'].attrs['NumPart_Total'] = numpart
            for group in ['PartType0']:
                if group not in out_file:
                    out_file.create_group(group)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    verbose = 2

    data_dir = f'/disk04/rad/sim/{model}/{wind}/'
    snapfile = f'{data_dir}snap_{model}_{snap}.hdf5'

    output_dir = f'/disk04/mrejus/sh/samples/'
    
    output_file = f'{output_dir}{model}_{wind}_{snap}.hdf5'
    particle_file = f'{output_dir}{model}_{wind}_{snap}_particle_selection.h5'

    #output_file = f'{output_dir}{model}_{wind}_{snap}_extras.hdf5'
    #particle_file = f'{output_dir}{model}_{wind}_{snap}_particle_selection_extras.h5'

    plist = np.array([])
    with h5py.File(particle_file, 'r') as f:
        for k in f.keys():
            if 'plist' in k:
                plist = np.append(plist, np.array(f[k][:], dtype='int'))

    plist = np.unique(np.sort(plist)).astype('int')

    numpart = np.zeros(6, dtype='int')
    numpart[0] = len(plist)

    prepare_out_file(snapfile, output_file, numpart)

    make_new_dataset(snapfile, output_file, plist, verbose)
