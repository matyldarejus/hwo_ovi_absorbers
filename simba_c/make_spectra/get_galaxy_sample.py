"""
Pick a galaxy sample based on star-formation rate.

Choose a random sample of 12 galaxies in each region, load in properties from the simulation
snapshot, and save to a sample file {sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5.

First step in the absorption analysis pipeline.
"""

import caesar
import yt
import numpy as np
import h5py
import sys

def belfiore_line(mstar, a=0.73, b=-7.7):
    # The definition of the SFMS from Belfiore+18 is:
    # log (SFR/Msun/yr) = 0.73 log (Mstar/Msun) - 7.33
    # With a scatter of sigma = 0.39 dex
    return mstar*a + b

def sfms_line(mstar, a=1., b=-10.8):
    # a cut of -10.8/yr as in other Simba work
    return mstar*a + b

def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask

# Randomly select ngals_each galaxies in each region
def choose_gals(gal_ids, ngals_each, seed=1):
    if seed is not None:
        np.random.seed(seed)
    ngals = len(gal_ids)
    chosen = np.random.choice(ngals, ngals_each, replace=False)
    return gal_ids[chosen]

model = sys.argv[1]
wind = sys.argv[2]
snap = sys.argv[3]

delta_m = 0.25
min_m = 10.
nbins_m = 6
mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
ngals_each = 12
nbins_ssfr = 3

sample_dir = f'/disk04/mrejus/sh/samples/' # save the samples to my disk04 directory
data_dir = f'/disk04/rad/sim/m100n1024/simba-c/' # access the simulation data necessary
sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
co = yt.utilities.cosmology.Cosmology(hubble_constant=sim.simulation.hubble_constant,
                                      omega_matter=sim.simulation.omega_matter,
                                      omega_lambda=sim.simulation.omega_lambda)
hubble = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/kpc')
redshift = sim.simulation.redshift
quench = quench_thresh(redshift)

gal_cent = np.array([i.central for i in sim.galaxies])
gal_sm = yt.YTArray([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))], 'Msun')
gal_sfr = yt.YTArray([sim.galaxies[i].sfr.in_units('Msun/yr') for i in range(len(sim.galaxies))], 'Msun/yr')
gal_ssfr = gal_sfr / gal_sm
gal_ssfr = np.log10(gal_ssfr.value + 1e-14)
gal_pos = yt.YTArray([sim.galaxies[i].pos.in_units('kpc/h') for i in range(len(sim.galaxies))], 'kpc/h')
gal_vels = yt.YTArray([sim.galaxies[i].vel.in_units('km/s') for i in range(len(sim.galaxies))], 'km/s')
gal_lgas = yt.YTArray([sim.galaxies[i].rotation['gas_L'].in_units('Msun*km*kpc/(h*s)') for i in range(len(sim.galaxies))], 'Msun*km*kpc/(h*s)')
gal_lbaryon = yt.YTArray([sim.galaxies[i].rotation['baryon_L'].in_units('Msun*km*kpc/(h*s)') for i in range(len(sim.galaxies))], 'Msun*km*kpc/(h*s)')
gal_sm = np.log10(gal_sm)
gal_recession = gal_pos.in_units('kpc')*hubble
gal_vgal_pos = gal_vels + gal_recession
gal_gas_frac = np.array([i.masses['gas'].in_units('Msun') /i.masses['stellar'].in_units('Msun') for i in sim.galaxies ])

halo_r200 = np.array([i.halo.virial_quantities['r200c'].in_units('kpc/h') for i in sim.galaxies]) 
r200_mask = halo_r200 > 0.

# empty arrays to store ngals_each simba galaxies per ssfr-mstar bin
gal_ids = np.ones(nbins_m*nbins_ssfr*ngals_each) * np.nan

sf_mask, gv_mask, q_mask = ssfr_type_check(quench, gal_ssfr)

for i in range(nbins_m):
    gal_id_range = range((i)*ngals_each*nbins_ssfr, (i+1)*ngals_each*nbins_ssfr)
    mass_mask = (gal_sm > mass_bins[i]) & (gal_sm < (mass_bins[i] + delta_m))

    # get the star forming galaxies:
    mask = gal_cent * mass_mask * sf_mask * r200_mask
    gals_possible = np.arange(len(sim.galaxies))[mask]
    if len(gals_possible) >= ngals_each:
        gal_ids[gal_id_range[0:ngals_each]] = choose_gals(gals_possible, ngals_each)
    else:
        print(f'Fewer than {ngals_each} galaxies in star forming bin {i}')
        gal_ids[gal_id_range[0:len(gals_possible)]] = gals_possible

    # get the green valley galaxies:
    mask = gal_cent * mass_mask * gv_mask * r200_mask
    gals_possible = np.arange(len(sim.galaxies))[mask]
    if len(gals_possible) >= ngals_each:
        gal_ids[gal_id_range[ngals_each:2*ngals_each]] = choose_gals(gals_possible, ngals_each)
    else:
        print(f'Fewer than {ngals_each} galaxies in green valley bin {i}')
        gal_ids[gal_id_range[ngals_each:ngals_each+len(gals_possible)]] = gals_possible

    # get the quenched galaxies:
    mask = gal_cent * mass_mask * q_mask * r200_mask
    gals_possible = np.arange(len(sim.galaxies))[mask]
    if len(gals_possible) >= ngals_each:
        gal_ids[gal_id_range[2*ngals_each:3*ngals_each]] = choose_gals(gals_possible, ngals_each)
    else:
        print(f'Fewer than {ngals_each} galaxies in quenched bin {i}')
        gal_ids[gal_id_range[2*ngals_each:2*ngals_each + len(gals_possible)]] = gals_possible

gal_ids = np.delete(gal_ids, np.arange(len(gal_ids))[np.isnan(gal_ids)])
gal_ids = gal_ids.astype('int')

halo_r200 = np.array([sim.galaxies[i].halo.virial_quantities['r200c'].in_units('kpc/h') for i in gal_ids])
halo_pos = np.array([sim.galaxies[i].halo.pos.in_units('kpc/h') for i in gal_ids])

with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'a') as hf:
    hf.create_dataset('gal_ids', data=np.array(gal_ids))
    hf.create_dataset('mass', data=np.array(gal_sm[gal_ids]))
    hf.create_dataset('ssfr', data=np.array(gal_ssfr[gal_ids]))
    hf.create_dataset('sfr', data=np.array(gal_sfr[gal_ids]))
    hf.create_dataset('fgas', data=np.array(gal_gas_frac[gal_ids]))
    hf.create_dataset('position', data=np.array(gal_pos[gal_ids]))
    hf.create_dataset('vgal_position', data=np.array(gal_vgal_pos[gal_ids]))
    hf.create_dataset('L_gas', data=np.array(gal_lgas[gal_ids]))
    hf.create_dataset('L_baryon', data=np.array(gal_lbaryon[gal_ids]))
    hf.create_dataset('halo_r200', data=np.array(halo_r200))
    hf.create_dataset('halo_pos', data=np.array(halo_pos))
    hf.attrs['pos_units'] = 'kpc/h'
    hf.attrs['mass_units'] = 'log Msun'
    hf.attrs['ssfr_units'] = 'log yr^-1'
    hf.attrs['sfr_units'] = 'log Msun/yr'
    hf.attrs['vel_units'] = 'km/s'

print(f'Galaxy samples for {model}, {wind}, {snap} generated.')    

