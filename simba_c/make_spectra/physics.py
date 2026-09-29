import pygad as pg
import numpy as np
import caesar
import h5py
from yt.utilities.cosmology import Cosmology


def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr


def get_total_column_density(logN, dlogN):
    totalN = np.sum(10**(logN))
    dtotalN = np.sqrt(np.sum(dlogN**2.) * (np.log(10.)**2.) * (totalN**2.))
    return convert_to_log(totalN, dtotalN)


def vel_to_wave(vel, lambda_rest, z, v_units='km/s'):
    c = pg.physics.c.in_units_of(v_units)
    return lambda_rest * (1.0 + z) * (vel / c + 1.)


def wave_to_vel(wave, lambda_rest, z, v_units='km/s'):
    c = pg.physics.c.in_units_of(v_units)
    return c * ((wave / lambda_rest) / (1.0 + z) - 1.0)


def tau_to_flux(tau):
    return np.exp(-np.clip(tau, -30, 30))


def wave_to_z(wave, lambda_rest):
    return (wave - lambda_rest) / lambda_rest


def get_redshift(wave, wave_rest): 
    return (wave - lambda_rest) / lambda_rest


def get_hubbles(model, wind, snap):
    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift
    co = Cosmology(hubble_constant=sim.simulation.hubble_constant, omega_matter=sim.simulation.omega_matter, omega_lambda=sim.simulation.omega_lambda)
    hubble_parameter = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/Mpc')
    hubble_constant = co.hubble_parameter(0).in_units('km/s/Mpc')
    return hubble_parameter, hubble_constant


def compute_dX(nlos, lines, path_lengths, redshift=0., hubble_parameter=68., hubble_constant=68.):

    idx = np.argmin(np.abs(path_lengths['redshifts'] - redshift))
    all_dX = np.zeros(len(lines))

    for i in range(len(lines)):
        dz = float(np.array(path_lengths[f'dz_{lines[i]}'])[idx]) * float(nlos)
        #print(nlos)
        #print(path_lengths[f'dz_{lines[i]}'])
        all_dX[i] = dz * (hubble_constant / hubble_parameter) * ((1 + redshift[0]) **2.)
    
    return all_dX


def create_path_length_file(vel_range, lines, redshifts, path_length_file):
    import pygad as pg
    lambda_rest = [float(pg.UnitQty(pg.analysis.absorption_spectra.lines[i]['l'])) for i in lines]
    c = float(pg.physics.c.in_units_of('km/s'))

    with h5py.File(path_length_file, 'a') as hf:
        hf.create_dataset('redshifts', data=np.array(redshifts))

    for i in range(len(lines)):
        path_length = np.zeros(len(redshifts))

        for j, z in enumerate(redshifts):

            lmin = lambda_rest[i] * (1+z)
            lmax = lambda_rest[i] * (1+z) * (1 + vel_range / c)

            zmin = wave_to_z(lmin, lambda_rest[i])
            zmax = wave_to_z(lmax, lambda_rest[i])

            path_length[j] = np.abs(zmax - zmin)

        with h5py.File(path_length_file, 'a') as hf:
             hf.create_dataset(f'dz_{lines[i]}', data=np.array(path_length))

