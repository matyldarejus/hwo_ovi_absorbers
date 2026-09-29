# Routines to call pygad to generate and save synthetic spectra.

import matplotlib.pyplot as plt
import numpy as np 
import pygad as pg
import gc
import os
import h5py
from pygadgetreader import readsnap
from physics import wave_to_vel, vel_to_wave
from utils import read_h5_into_dict 

def t_elapsed(): 
    return np.round(time.time()-TINIT,2)


def write_spectrum(spec_name, line, los, lambda_rest, gal_vel_pos, redshift, snr, spectrum):

    if len(los) == 2: 
        los = np.append(np.array(los), -1.0)  # assumes if only 2 values are provided, they are (x,y), so we add -1 for z. 

    with h5py.File(spec_name, "w") as hf:
        lam0 = hf.create_dataset("lambda_rest", data=lambda_rest)
        lam0.attrs["ion_name"] = line  # store line name as attribute of rest wavelength
        hf.create_dataset("gal_velocity_pos", data=np.array(gal_vel_pos))
        hf.create_dataset("LOS_pos", data=np.array(los))
        hf.create_dataset("redshift", data=redshift)
        hf.create_dataset("snr", data=snr)
        for k in spectrum.keys():
            hf.create_dataset(k, data=np.array(spectrum[k]))


def write_line_list(spectrum, spec_file):
    
    with h5py.File(spec_file, 'a') as hf:
        if 'lines' in hf.keys():
            del hf['lines']
        lines = hf.create_group("line_list")
        lines.create_dataset("region", data=np.array(spectrum['line_list']['region']))
        lines.create_dataset("N", data=np.array(spectrum['line_list']['N']))
        lines.create_dataset("dN", data=np.array(spectrum['line_list']['dN']))
        lines.create_dataset("b", data=np.array(spectrum['line_list']['b']))
        lines.create_dataset("db", data=np.array(spectrum['line_list']['db']))
        lines.create_dataset("l", data=np.array(spectrum['line_list']['l']))
        lines.create_dataset("dl", data=np.array(spectrum['line_list']['dl']))
        lines.create_dataset("EW", data=np.array(spectrum['line_list']['EW']))
        lines.create_dataset("Chisq", data=np.array(spectrum['line_list']['Chisq']))


def generate_pygad_spectrum(s, los, line, lambda_rest, gal_vel_pos, periodic_vel, pixel_size, snr, spec_name, LSF, fit_contin=False, min_restr_column=5):
    if os.path.isfile(f'{spec_name}.h5'):
        return

    print('Generating pygad spectrum for ' + line)

    c = pg.physics.c.in_units_of('km/s') 
    H = s.cosmology.H(s.redshift).in_units_of('(km/s)/Mpc', subs=s)
    box_width = s.boxsize.in_units_of('Mpc', subs=s)
    vbox = H * box_width
    if periodic_vel:
        v_limits = [-0.5*vbox, 1.5*vbox]
    else:
        v_limits = [0, vbox]
    total_v = np.sum(np.abs(v_limits))
    Nbins = int(np.rint(total_v / pixel_size))

    spectrum = {}
    spectrum['taus'], spectrum['col_density'], spectrum['phys_density'], spectrum['temperature'], spectrum['metallicity'], spectrum['vpec'], v_edges, spectrum['restr_column'] = \
                pg.analysis.absorption_spectra.mock_absorption_spectrum_of(s, los, line, v_limits, Nbins=(1+int(periodic_vel))*Nbins, return_los_phys=True)
    spectrum['velocities'] = 0.5 * (v_edges[1:] + v_edges[:-1])
    
    if periodic_vel:
        npix_periodic = int( vbox / (max(spectrum['velocities'])-min(spectrum['velocities'])) * len(spectrum['taus']) )
        print('periodically wrapping optical depths with %d pixels'%npix_periodic)
        for i in range(0,len(spectrum['taus'])):
            if spectrum['velocities'][i] < -vbox: spectrum['taus'][i+npix_periodic] += spectrum['taus'][i]
            if spectrum['velocities'][i] > 0: spectrum['taus'][i-npix_periodic] += spectrum['taus'][i]
        # clip spectrum to original vbox
        i0 = (np.abs(spectrum['velocities'])).argmin()
        i1 = (np.abs(spectrum['velocities']-vbox)).argmin()
        spectrum['velocities'] = spectrum['velocities'][i0:i1]
        spectrum['taus'] = spectrum['taus'][i0:i1]
        spectrum['col_density'] = spectrum['col_density'][i0:i1]
        spectrum['phys_density'] = spectrum['phys_density'][i0:i1]
        spectrum['temperature'] = spectrum['temperature'][i0:i1]
        spectrum['metallicity'] = spectrum['metallicity'][i0:i1]
        spectrum['vpec'] = spectrum['vpec'][i0:i1]

    spectrum['wavelengths'] = lambda_rest * (s.redshift + 1) * (1 + spectrum['velocities'] / c)
    spectrum['metallicity'] = np.log10(np.where(spectrum['metallicity']<1.e-6,1.e-6,spectrum['metallicity']))

    sigma_noise = 1.0/snr
    spectrum['noise'] = np.random.normal(0.0, sigma_noise, len(spectrum['wavelengths']))
    noise_vector = np.asarray([sigma_noise] * len(spectrum['noise']))
    spectrum['fluxes'] = np.exp(-np.array(spectrum['taus'])) + spectrum['noise']

    if not line == 'MgII2796':
        if LSF is not None and spectrum['wavelengths'][0] > 900:
            spectrum['fluxes'],noise_vector = pg.analysis.absorption_spectra.apply_LSF(spectrum['wavelengths'], spectrum['fluxes'], noise_vector, grating=LSF)
    else:
        from astropy.convolution import convolve, Gaussian1DKernel
        fwhm = 6. / pixel_size # 6km s^-1, in pixels
        gauss_kernel = Gaussian1DKernel(stddev=fwhm / 2.355)
        spectrum['fluxes'] = convolve(spectrum['fluxes'], gauss_kernel, boundary="wrap")
        noise_vector = convolve(noise_vector, gauss_kernel, boundary="wrap")
    
    if fit_contin:
        spectrum['continuum'] = pg.analysis.absorption_spectra.fit_continuum(spectrum['wavelengths'], spectrum['fluxes'], noise_vector, order=0, sigma_lim=1.5)
        spectrum['fluxes'] = spectrum['fluxes']/spectrum['continuum']

    spectrum['n90_restr_column'] = check_restr_column(s.filename, los, spectrum['restr_column'])
    if spectrum['n90_restr_column'] < min_restr_column:
        print(f'WARNING: fewer than {min_restr_column} gas elements contribute 90% of the total column density.')
    del spectrum['restr_column']

    total_EW = np.sum(1. - spectrum['fluxes']) * pixel_size * lambda_rest / c
    print(r'Npix with abs: %d  EW: %g Angstrom' % (len(spectrum['fluxes'][1. - spectrum['fluxes'] > 3. / snr]), total_EW))
    write_spectrum(f'{spec_name}.h5', line, los, lambda_rest, gal_vel_pos, s.redshift, snr, spectrum)

    return


def get_los_particles(los, gas_pos, hsml):

    x_dist = np.abs(los[0] - gas_pos[:, 0])
    y_dist = np.abs(los[1] - gas_pos[:, 1])
    hyp_sq = x_dist**2 + y_dist**2
    dist_mask = hyp_sq < hsml**2
    partids_los = np.arange(len(hsml))[dist_mask]
    return partids_los


def check_restr_column(snapfile, los, restr_column, f_total_column=0.9):
    
    hsml = readsnap(snapfile, 'SmoothingLength', 'gas', suppress=1, units=1)  # in kpc/h, comoving
    gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) # in kpc/h, comoving
    los_particles = get_los_particles(los, gas_pos, hsml)

    los_restr_column = restr_column[los_particles]
    total_column = np.sum(los_restr_column)
    restr_cumsum = np.cumsum(np.sort(los_restr_column)[::-1])
    try:
        i = next(i for i, x in enumerate(restr_cumsum) if x >= f_total_column * total_column)
        return i
    except:
        return -9999
