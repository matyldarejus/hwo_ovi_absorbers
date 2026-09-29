"""
Run Pygad spectrum fitter on each line-of-sight

This is basically just an orchestrator for generate_spectra.

Ran for each galaxy individually, hence best ran via a bash file (sub_pipeline).
"""


# Main pipeline script to run pygad on each LOS
# Run using sub_line_pipeline.sh and sub_pipeline.sh

import sys
import os
import numpy as np
import h5py
import re
import yt
from yt.units.yt_array import YTArray, YTQuantity
import pygad as pg
from generate_spectra import generate_pygad_spectrum

# Set some spectrum parameters:
sqrt2 = np.sqrt(2.)
snr = 30.
vel_range = 600. # km/s
pixel_size = 2.5 # km/s
periodic_vel = True
#LSF = None
LSF = 'STIS_E140M' # FWHM 6km - spreads out absorption line over several pixels
fit_contin = True
delta_fr200 = 0.25
min_fr200 = 0.25
nbins_fr200 = 5
fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

model = sys.argv[1]
wind = sys.argv[2]
snap = sys.argv[3]
num = int(sys.argv[4])
line = sys.argv[5]
lambda_rest = float(re.findall(r'\d+', line)[0])

snapfile = f'/disk04/mrejus/sh/samples/{model}_{wind}_{snap}.hdf5'
s = pg.Snapshot(snapfile)

sample_dir = f'/disk04/mrejus/sh/samples/'
save_dir =  f'/disk04/mrejus/sh/normal/{model}_{wind}_{snap}_hm12/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:

    gal_id = sf['gal_ids'][:][num]
    # we can't have the line of sight as a pygad UnitArr because it can't convert between kpc/h and ckpc/h_0
    # so instead we convert to default units of s['pos']
    # hence kpc/h and the factor of (1+z) is necessary
    gal_pos = (sf['position'][:]  *  (1.+s.redshift))[num]
    gal_vel_pos = sf['vgal_position'][:][num][2]
    gal_r200 = sf['halo_r200'][:][num] * (1.+s.redshift) # already in kpc/h, factor of 1+z for comoving


print(f'Generating spectra for sample galaxy {gal_id} {gal_pos} {gal_r200}')
gal_name = f'sample_galaxy_{gal_id}_'

for j in range(len(fr200)):

    rho = gal_r200 * fr200[j]

    spec_name = f'{gal_name}{line}_0_deg_{fr200[j]}r200'
    los = gal_pos[:2].copy(); los[0] += rho
    print('In kpc/h: ' + str(los))
    generate_pygad_spectrum(s, los, line, lambda_rest, gal_vel_pos, periodic_vel, pixel_size, snr, f'{save_dir}{spec_name}', LSF=LSF, fit_contin=fit_contin)

    spec_name = f'{gal_name}{line}_45_deg_{fr200[j]}r200'
    los = gal_pos[:2].copy(); los[0] += (rho / sqrt2); los[1] += (rho / sqrt2)
    generate_pygad_spectrum(s, los, line, lambda_rest, gal_vel_pos, periodic_vel, pixel_size, snr, f'{save_dir}{spec_name}', LSF=LSF, fit_contin=fit_contin)

    spec_name = f'{gal_name}{line}_90_deg_{fr200[j]}r200'
    los = gal_pos[:2].copy(); los[1] += rho
    generate_pygad_spectrum(s, los, line, lambda_rest, gal_vel_pos, periodic_vel, pixel_size, snr, f'{save_dir}{spec_name}', LSF=LSF, fit_contin=fit_contin)

    spec_name = f'{gal_name}{line}_135_deg_{fr200[j]}r200'
    los = gal_pos[:2].copy(); los[0] -= (rho / sqrt2); los[1] += (rho / sqrt2)
    generate_pygad_spectrum(s, los, line, lambda_rest, gal_vel_pos, periodic_vel, pixel_size, snr, f'{save_dir}{spec_name}', LSF=LSF, fit_contin=fit_contin)

    spec_name = f'{gal_name}{line}_180_deg_{fr200[j]}r200'
    los = gal_pos[:2].copy(); los[0] -= rho
    generate_pygad_spectrum(s, los, line, lambda_rest, gal_vel_pos, periodic_vel, pixel_size, snr, f'{save_dir}{spec_name}', LSF=LSF, fit_contin=fit_contin)

    spec_name = f'{gal_name}{line}_225_deg_{fr200[j]}r200'
    los = gal_pos[:2].copy(); los[0] -= (rho / sqrt2); los[1] -= (rho / sqrt2)
    generate_pygad_spectrum(s, los, line, lambda_rest, gal_vel_pos, periodic_vel, pixel_size, snr, f'{save_dir}{spec_name}', LSF=LSF, fit_contin=fit_contin)

    spec_name = f'{gal_name}{line}_270_deg_{fr200[j]}r200'
    los = gal_pos[:2].copy(); los[1] -= rho
    generate_pygad_spectrum(s, los, line, lambda_rest, gal_vel_pos, periodic_vel, pixel_size, snr, f'{save_dir}{spec_name}', LSF=LSF, fit_contin=fit_contin)

    spec_name = f'{gal_name}{line}_315_deg_{fr200[j]}r200'
    los = gal_pos[:2].copy(); los[0] += (rho / sqrt2); los[1] -= (rho / sqrt2)
    generate_pygad_spectrum(s, los, line, lambda_rest, gal_vel_pos, periodic_vel, pixel_size, snr, f'{save_dir}{spec_name}', LSF=LSF, fit_contin=fit_contin)

