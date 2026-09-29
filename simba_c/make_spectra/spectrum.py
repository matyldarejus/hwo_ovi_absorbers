from pygad.analysis.absorption_spectra import lines, Gaussian, Lorentzian, Voigt, line_profile
from pygad.units import Unit, UnitArr, UnitQty, UnitScalar
import pygad.environment as environment
import os
import sys

import numpy as np
import os
import sys
import h5py
import matplotlib.pyplot as plt
import pygad as pg
from physics import wave_to_vel, vel_to_wave, tau_to_flux
from utils import read_h5_into_dict
from scipy import signal
plt.rcParams['text.usetex'] = False
import scipy

class Spectrum(object):


    def __init__(self, spectrum_file, **kwargs):

        self.spectrum_file = spectrum_file
        for key in kwargs:
            setattr(self, key, kwargs[key])
        data = read_h5_into_dict(self.spectrum_file)
        for key in data:
            setattr(self, key, data[key])
        del data
        f = h5py.File(spectrum_file, 'r')
        #self.gal_velocity_pos = np.array(f['galaxy_properties/vlos'])
        self.gal_velocity_pos = np.array(f['gal_velocity_pos'])
        self.redshift = np.array(f['redshift'])
        self.lambda_rest = np.array(f['lambda_rest'])
        self.velocities = np.array(f['velocities'])
        self.wavelengths = self.lambda_rest * (self.redshift + 1) * (1 + self.velocities / 3e5)
        self.fluxes = np.array(f['fluxes'])
        self.noise = np.array(f['noise'])
        self.continuum = np.ones(len(self.velocities))


    def get_initial_window(self, vel_range, v_central=None, v_boxsize=10000.):

        # get the portion of the CGM spectrum that we want to fit

        def _find_nearest(array, value):
            return np.abs(array - value).argmin()

        if v_central is None:
            v_central = self.gal_velocity_pos

        # get the velocity start and end positions
        dv = self.velocities[1] - self.velocities[0]
        v_start = v_central - vel_range
        v_end = v_central + vel_range
        N = int((v_end - v_start) / dv)

        # get the start and end indices
        if v_start < 0.:
            v_start += v_boxsize
        i_start = _find_nearest(self.velocities, v_start)
        i_end = i_start + N

        return i_start, i_end, N


    def extend_to_continuum(self, i_start, i_end, N, contin_level=None):

        # from the initial velocity window, extend the start and end back to the level of the continuum of the input spectrum/

        if contin_level is None:
            contin_level = self.continuum[0]

        continuum = False
        while not continuum:
            _flux = self.fluxes.take(i_start, mode='wrap')
            if np.abs(_flux - contin_level) / contin_level > 0.02:
                i_start -= 1
                N += 1
            else:
                continuum = True

        continuum = False
        while not continuum:
            _flux = self.fluxes.take(i_end, mode='wrap')
            if np.abs(_flux - contin_level) / contin_level > 0.02:
                i_end += 1
                N += 1
            else:
                continuum = True

        return i_start, i_end, N
   

    def buffer_with_continuum(self, waves, flux, nbuffer=50, snr_default=30.):

        # add a buffer to either end of the velocity window at the continuum level to aid the voigt fitting.

        if hasattr(self, 'snr'):
            snr = self.snr
        else:
            snr = snr_default
        dl = waves[1] - waves[0]
        l_start = np.arange(waves[0] - dl*nbuffer, waves[0], dl)
        l_end = np.arange(waves[-1]+dl, waves[-1] + dl*(nbuffer+1), dl)
        
        waves = np.concatenate((l_start, waves, l_end))

        sigma_noise = 1./snr
        new_noise = np.random.normal(0.0, sigma_noise, 2*nbuffer)
        flux = np.concatenate((tau_to_flux(np.zeros(nbuffer)) + new_noise[:nbuffer], flux, tau_to_flux(np.zeros(nbuffer)) + new_noise[nbuffer:]))
        
        return waves, flux

    def prepare_spectrum(self, vel_range, do_continuum_buffer=True, nbuffer=50, snr_default=30):

        # cut out the portion of the spectrum that we want within some velocity range, making sure the section we cut out 
        # goes back up to the conintuum level (no dicontinuities)

        i_start, i_end, N = self.get_initial_window(vel_range)
        i_start, i_end, N = self.extend_to_continuum(i_start, i_end, N)

        # cope with spectra that go beyond the left hand edge of the box (periodic wrapping)
        if i_start < 0:
            i_start += len(self.wavelengths)
            i_end += len(self.wavelengths)

        # extract the wavelengths and fluxes for fitting
        self.waves_fit = self.wavelengths.take(range(i_start, i_end), mode='wrap')
        self.fluxes_fit = self.fluxes.take(range(i_start, i_end), mode='wrap')

        # check if the start and end wavelengths go over the limits of the box
        i_wrap = len(self.wavelengths) - i_start
        wave_boxsize = self.wavelengths[-1] - self.wavelengths[0]
        dl = self.wavelengths[1] - self.wavelengths[0]
        if i_wrap < N:
            # spectrum wraps, i_wrap is the first index of the wavelengths that have been moved to the left side of the box
            self.waves_fit[i_wrap:] += wave_boxsize + dl
            # then for any fitted lines with position outwith the right-most box limits: subtract dl + wave_boxsize

        # add a buffer of continuum to either side to help the voigt fitter identify where to fit
        if do_continuum_buffer is True:
            self.waves_fit, self.fluxes_fit = self.buffer_with_continuum(self.waves_fit, self.fluxes_fit, nbuffer=nbuffer)

        # get the noise level
        if hasattr(self, 'snr'):
            snr = self.snr
        else:
            snr = snr_default
        self.noise_fit = np.asarray([1./snr] * len(self.fluxes_fit))


    def fit_spectrum_old(self, vel_range=600., nbuffer=20):

        # old fitting routine - not required

        contin_level = self.continuum[0]
        self.extend_to_continuum(vel_range, contin_level)

        self.line_list = pg.analysis.fit_profiles(self.ion_name, self.wavelengths[self.vel_mask], self.fluxes[self.vel_mask], self.noise[self.vel_mask],
                                                  chisq_lim=2.5, max_lines=10, logN_bounds=[12,17], b_bounds=[2,100], mode='Voigt')
    
        self.line_list['v'] = wave_to_vel(self.line_list['l'], self.lambda_rest, self.redshift)

        outwith_vel_mask = ~((self.line_list['v'] > self.gal_velocity_pos - vel_range) & (self.line_list['v'] < self.gal_velocity_pos + vel_range))

        for k in self.line_list.keys():
            self.line_list[k] = np.delete(self.line_list[k], outwith_vel_mask)


    def fit_periodic_spectrum(self):

        # the fitting approach for periodic spectra, i.e. those which span the length of the Simba volume

        wrap_flux, wrap_noise, wrap_start = pg.analysis.periodic_wrap(self.wavelengths, self.fluxes, self.noise)
        self.line_list = pg.analysis.fit_profiles(self.ion_name, self.wavelengths, wrap_flux, wrap_noise,
                                         chisq_lim=2.5, max_lines=10, logN_bounds=[12,17], b_bounds=[2,100], mode='Voigt')
        self.line_list['l'] = pg.analysis.periodic_unwrap_wavelength(self.line_list['l'], self.wavelengths, wrap_start)
        self.line_list['v'] = wave_to_vel(self.line_list['l'], self.lambda_rest, self.redshift)

        outwith_vel_mask = ~((self.line_list['v'] > self.gal_velocity_pos - vel_range) & (self.line_list['v'] < self.gal_velocity_pos + vel_range))

        for k in self.line_list.keys():
            self.line_list[k] = np.delete(self.line_list[k], outwith_vel_mask)

        # find the temperature corresponding to the l wavelength line widths - not sure if this is needed / works 
        if 'l' in self.line_list:
            print("Finding line temperatures...")
            line_wav = self.line_list['l']

            all_wav = self.wavelengths

            def _find_nearest(array, value):
                return np.abs(array - value).argmin()
            
            idx = [_find_nearest(all_wav, lw) for lw in line_wav]
            temp = self.temperature[idx]
            print(f"Line temperatures: {temp}")
            self.line_list['t'] = temp



    def get_tau_model(self):

        # compute the total optical depth of the model from the individual lines

        self.tau_model = np.zeros(len(self.wavelengths))
        for i in range(len(self.line_list["N"])):
            p = np.array([self.line_list["N"][i], self.line_list["b"][i], self.line_list["l"][i]])
            self.tau_model += pg.analysis.model_tau(self.ion_name, p, self.wavelengths, 'Voigt')

    
    def get_fluxes_model(self):

        # compute the total flux from the individual lines

        self.get_tau_model()
        self.fluxes_model = tau_to_flux(self.tau_model)


    def write_line_list(self):

        # save the components of the fit in h5 format to the original input file

        with h5py.File(self.spectrum_file, 'a') as hf:
            print('Creating line list')
            if 'line_list' in hf.keys():
                print('Deleting old line list')
                del hf['line_list']
            elif 'lines' in hf.keys():
                del hf['lines']
            line_list = hf.create_group("line_list")
            for k in self.line_list.keys():
                line_list.create_dataset(k, data=np.array(self.line_list[k]))

    def plot_fit(self, ax=None, vel_range=600., filename=None):

        # plot the results :)

        if ax is None:
            fig, ax = plt.subplots()

        x_val = self.wavelengths

        ax.plot(x_val, self.fluxes, label='data', c='tab:grey', lw=2, ls='-')

        self.get_fluxes_model()
        for i in range(len(self.line_list['N'])):
            p = np.array([self.line_list['N'][i], self.line_list['b'][i], self.line_list['l'][i]])
            print('Plotting',i,p)
            _tau_model = pg.analysis.model_tau(self.ion_name, p, self.wavelengths)
            ax.plot(x_val, tau_to_flux(_tau_model), alpha=0.5, lw=1, ls='--', label='%g %g'%(self.line_list['N'][i],self.line_list['b'][i]))

        ax.plot(x_val, self.fluxes_model, label='model', c='tab:pink', ls='--', lw=2)

        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(x_val[0], x_val[-1])
        #ax.set_xlim(0, self.gal_velocity_pos +vel_range)
        #ax.set_xlim(max(self.gal_velocity_pos-2*vel_range, 0), min(self.gal_velocity_pos+2*vel_range, self.velocities[-1]))
        ax.legend(loc='best',fontsize=8)
        
        chisq = np.around(np.unique(self.line_list['Chisq']), 2)
        chisq = [str(i) for i in chisq]
        plt.title(r'$\chi^2_r = {x}$'.format(x = ', '.join(chisq) ))
        
        if filename == None:
            filename = self.spectrum_file.split('/')[-1].replace('.h5', '.png')
        plt.savefig(f'/disk04/mrejus/sh/results/plots/spec//{filename}')
        plt.show()
        #plt.savefig('../figures/spec_gal_1.png')
        plt.close()

    def main(self, vel_range, do_continuum_buffer=True, nbuffer=50, 
             snr_default=30., chisq_unacceptable=25, chisq_asym_thresh=-3., 
             do_prepare=True, do_regions=False, do_fit=True, write_lines=True, plot_fit=False):
 
        self.chisq_asym_thresh = chisq_asym_thresh

        # prepare the portion of the spectrum to fit
        # extract from full spectrum, wrap periodically, buffer with a continuum, set the noise level for fitting
        if do_prepare:
            print('Preparing spectrum vel_range +/-',vel_range,' km/s')
            self.prepare_spectrum(vel_range, do_continuum_buffer=True, nbuffer=50, snr_default=30.,)

        # to identify the region boundaries only:
        if do_regions:
            if do_prepare is not True:
                print('Spectrum not prepared; set do_prepare=True and retry :)')
                return
            else:
                self.line_list = {}
                self.regions_l, self.regions_i, _ = pg.analysis.find_regions(self.waves_fit, self.fluxes_fit, self.noise_fit, min_region_width=2, extend=True)
                print(f'These are the regions {self.regions_l}')
                self.line_list['region'] = np.arange(len(self.regions_l))

        # to perform the voigt fitting:
        if do_fit:
            if self.ion_name == 'H1215':
                logN_bounds = [12, 19]
                b_bounds=[5,200]
            else:
                logN_bounds = [10, 17]
            b_bounds=[1,100]
            print('Doing fitting for ion %s, lambda range=[%.2f,%.2f]' % (self.ion_name, self.wavelengths[0], self.wavelengths[-1]))
            self.line_list = fit_profiles_sat(self.ion_name, self.waves_fit, self.fluxes_fit, self.noise_fit, self.wavelengths, self.temperature,
                                                      chisq_lim=2.5,
                                                      max_lines=10, logN_bounds=logN_bounds, 
                                                      b_bounds=b_bounds, mode='Voigt')
            '''
            self.line_list = pg.analysis.fit_profiles(self.ion_name, self.wavelengths[self.vel_mask], self.fluxes[self.vel_mask], self.noise[self.vel_mask],
                                                  chisq_lim=2.5, max_lines=10, logN_bounds=[12,17], b_bounds=[3,100], mode='Voigt')
            self.line_list = pg.analysis.fit_profiles(self.ion_name, self.waves_fit, self.fluxes_fit, self.noise_fit,
                                                      chisq_lim=2.5, chisq_unacceptable=chisq_unacceptable, 
                                                      chisq_asym_thresh=chisq_asym_thresh, 
                                                      max_lines=10, logN_bounds=logN_bounds, 
                                                      b_bounds=b_bounds, mode='Voigt')
            '''
            
            wave_boxsize = self.wavelengths[-1] - self.wavelengths[0]
            dl = self.wavelengths[1] - self.wavelengths[0]
            
            # adjust the output lines to cope with wrapping
            for i in range(len(self.line_list['l'])):
                if self.line_list['l'][i] > self.wavelengths[-1]:
                    self.line_list['l'][i]  -= (wave_boxsize + dl)
                elif self.line_list['l'][i] < self.wavelengths[0]:
                    self.line_list['l'][i] += (wave_boxsize + dl)

        # keep the fit, save to the original spectrum file
        #print(self.line_list)
        if write_lines:
            self.write_line_list()

        if plot_fit:
            self.plot_fit()
    




def fit_profiles_sat(
    line,
    l,
    flux,
    noise,
    wavelengths,
    temperatures,
    chisq_lim=2.5,
    max_lines=10,
    mode="Voigt",
    logN_bounds=[8,20],
    b_bounds=[1, 300],
):
    """
    Fit Voigt/other profiles to the given spectrum.  Begins with one
    line, then adds lines until desired chi-sq is achieved.

    Args:
        line (str):         The line to fit as listed in
                            `analysis.absorption_spectra.lines`, e.g. 'HI1215'.
        l (array-like):     The wavelengths of the input spectrum to fit.
        flux (array-like):  The normalized flux at the given wavelengths,
                            i.e. the spectrum to fit.
        noise (array-like): The normalized 1-sigma noise vector at the given
                            wavelengths.  Must always be >0.
        chisq_lim (float):  Number of sigma below which chi-sq is considered
                            to be a "good fit" and no more lines are added.
                            If <0, then the value used is abs(chisq_lim)+0.1*n_lines,
                            where n_lines is the number of lines for that trial.
        max_lines (int):    Maximum number of lines allowed in a given detection
                            region, after which the fit declared done regardless of chisq.
                            If limit is hit, this may result in a poor fit.
        mode (str):         Type of line profile: Gaussian/Lorentzian/Voigt;
                            see absorption_spectra.line_profile().
        logN_bounds (list): Initial log(column density) is restricted to this range
        b_bounds (list):    Initial line width is restricted to this range (km/s)


    Returns:
        profiles:    Dictionary of [N, dN, b, db, l, dl, EW] of best-fit
                     profiles.
        tau_model:   Optical depths of best-fit model.

    """

    verbose = (environment.verbose >= environment.VERBOSE_QUIET)
    np.set_printoptions(formatter={'float': '{:.4f}'.format})

    #plt.plot(l,flux)
    if isinstance(line, str):
        line = lines[line]
    l0 = line["l"]
    if isinstance(l, np.ndarray) or l.units in [
        1,
        None,
    ]:  # set units of l to Angstrom if none supplied
        l = UnitArr(l, "Angstrom")

    def _tau_to_flux(tau):  # return flux from tau, avoiding over/underflow
        return np.exp(-np.clip(tau, -50, 50))

    def _chisq_asym(p, l, flux, noise, mode):  # reduced chisq, suppressing saturated regions
        #chisq_asym = -1.
        model_flux = _tau_to_flux(model_tau(line, p, l, mode))
        dx_array = (flux - model_flux) / noise
        #dx_array = np.where(dx_array > chisq_asym, dx_array, dx_array - (dx_array - chisq_asym)**2)
        dx_array = np.where(flux < 0., 0., dx_array)
        return np.sum(dx_array * dx_array) / np.count_nonzero(dx_array)

    def _chisq(p, l, flux, noise, mode):  # reduced chisq
        model_flux = _tau_to_flux(model_tau(line, p, l, mode))
        dx_array = (flux - model_flux) / noise
        #dx_array = np.where(flux < abs(noise), 0., dx_array)
        return np.sum(dx_array * dx_array) / np.count_nonzero(dx_array)

    def _add_line(p, bnd, l, flux, noise, l0, mode, i_line=None, grow_line=True):  # adds N, b, l for a new line
        if len(p) == 0:
            resid = flux
        else:
            resid = 1.0 + flux - _tau_to_flux(model_tau(line, p, l, mode))  # residual spectrum
        l_bounds = [l[1], l[-2]]

        if grow_line:
            # Grow line to max (N,b) allowed given the residual
            n_guess, b_guess, l_guess = _grow_line(l, flux, noise, resid, l0, mode, i_line=i_line)
        else:
            # Make an educated guess at the new line parameters
            b_guess = (
                (l_bounds[1] - l_bounds[0]) / float(l0) * 3.0e5 / 5.0
            )  # first guess at b
            b_guess = max(2 * b_bounds[0], 0.5 * min(b_bounds[1], b_guess))
            n_guess = 14.0 - resid[np.argmin(resid)]  # first guess at logN
            l_guess = l[np.argmin(resid)]

        # append line
        p = np.append(p, n_guess)  # rough guess of logN
        p = np.append(p, b_guess)  # first guess of b
        p = np.append(p, l_guess)  # add line @min of residual flux

        # append bounds
        n_bounds = [n_guess-0.5, n_guess+0.5]
        b_bounds = [b_guess*0.5, b_guess*2]
        if len(bnd) == 0:
            bnd = np.array([n_bounds])
        else:
            bnd = np.append(bnd, np.array([n_bounds]), axis=0)
        bnd = np.append(bnd, np.array([b_bounds]), axis=0)
        bnd = np.append(bnd, np.array([l_bounds]), axis=0)
        return p, bnd

    def _grow_line(l, flux, noise, resid, l0, mode, i_line=None, floor_sigma=1.5, smooth_sigma=1., unsat_sigma=3.):  # adds N, b, l for a new line at l by growing N, b
        # compute location of new line, if pixel value not specified in i_line
        if smooth_sigma > 0.:
            smoothed = scipy.ndimage.gaussian_filter1d(resid, smooth_sigma)
        else:
            smoothed = resid
        if i_line is None:
            i_line = np.argmin(smoothed)

        l_line = l[i_line]
        if resid[i_line] < np.min(abs(noise)):
            # if saturated use the center of the saturated region 
            i_lo = i_line
            while resid[i_lo] < unsat_sigma * noise[i_lo] and i_lo > 0: i_lo -= 1
            i_hi = i_line
            while resid[i_hi] < unsat_sigma * noise[i_hi] and i_hi < len(l)-1: i_hi += 1
            i_line = int(0.5 * (i_lo+i_hi))
            l_line = l[i_line]
            N_lim  = logN_bounds[1]
            b_lim = 2.*min(abs(l_line-l[i_lo]),abs(l[i_hi]-l_line)) * 3.e5 / float(l0)
        else:
            # if unsaturated use min of smoothed residual
            N_lim = 15.0
            fdec_bottom = 1.-resid[i_line]
            i_lo = i_line
            while 1.-resid[i_lo] < 0.5 * fdec_bottom and i_lo > 0: i_lo -= 1
            i_hi = i_line
            while 1.-resid[i_hi] < 0.5 * fdec_bottom and i_hi < len(l)-1: i_hi += 1
            b_lim = 4.*min(abs(l_line-l[i_lo]),abs(l[i_hi]-l_line)) * 3.e5 / float(l0)
        b_lim = min(max(b_lim, max(b_bounds[0],20)), b_bounds[1])

        # Set floor which model cannot go below
        floor = smoothed - floor_sigma * noise
        # set up range in N,b
        N_range = np.linspace(start=logN_bounds[0], stop=N_lim, num=40)
        b_range = np.linspace(start=np.log10(b_bounds[0]), stop=np.log10(b_lim), num=40)
        b_range = 10**b_range
        # incremeent N,b until model goes below floor
        N_min = logN_bounds[0]
        p_allowed = np.array([logN_bounds[0], b_bounds[0], l_line])
        chisq = [1.e20]
        for bpar in b_range:
            for Ncol in N_range:
                if Ncol < N_min:
                    continue
                p_trial = np.array([Ncol, bpar, l_line])
                model = _tau_to_flux(model_tau(line, p_trial, l, mode))
                diff = model-floor
                # Save the largest line that satisfies the condition model>floor everywhere 
                if np.any(diff<0):
                    #print('Cannot add:',bpar,Ncol,floor[diff<0], diff[diff<0], l_line,l[diff<0],l[0],l[-1])
                    break
                else:
                    p_allowed = np.append(p_allowed, p_trial)
                    # compute chi-sq just near this line
                    i_lo = np.argmin(model)
                    while (1.-model[i_lo]) > 0.5 * (1.-model[i_line]) and i_lo > 0: i_lo -= 1
                    i_hi = np.argmin(model)
                    while (1.-model[i_hi]) > 0.5 * (1.-model[i_line]) and i_hi < len(model)-2: i_hi += 1
                    dx_array = (resid[i_lo:i_hi+1] - model[i_lo:i_hi+1]) / noise[i_lo:i_hi+1]
                    chi2 = np.sum(dx_array * dx_array) / np.count_nonzero(dx_array)
                    chisq.append(chi2)
                    #print('Could add:',bpar,Ncol,l_line,chisq[-1])
        i_p = np.argmin(np.array(chisq))  # from all allowed lines, choose one with lowest chisq
        return p_allowed[3*i_p], p_allowed[3*i_p+1], l_line

    def _maxiter(n, nmax):
        if n <= 5: return 100
        else: return max(50, 50+(nmax-n)*10)

    # identify independent regions to fit within the spectrum
    regions_l, regions_i, _ = find_regions(l, flux, noise, min_region_width=2)

    # dicts to store results
    line_list = {
        "region": np.array([]),
        "l": np.array([]),
        "dl": np.array([]),
        "b": np.array([]),
        "db": np.array([]),
        "N": np.array([]),
        "dN": np.array([]),
        "EW": np.array([]),
        "Chisq": np.array([]),
        "t": np.array([])   # temperature for each line
    }

    # loop over regions
    from scipy.optimize import minimize
    
    sat_regions = False
    
    for ireg in range(len(regions_l)):
    
        params = []
        bounds = []
        n_lines = 0
        best_nlines = 1
        chisq_old = 1.0e20
        chisq_accept = abs(chisq_lim)
        l_reg = l[regions_i[ireg, 0] : regions_i[ireg, 1]]
        f_reg = flux[regions_i[ireg, 0] : regions_i[ireg, 1]]
        n_reg = noise[regions_i[ireg, 0] : regions_i[ireg, 1]]
        #plt.plot(l_reg,f_reg)


        ###Saturated Region Detection
        regions_l_sat, regions_i_sat, bounding_i = find_regions_sat(l_reg,f_reg,n_reg,min_region_width=15)
        bounding_i = [] # don't use sat region detection
        
        params_reg = []
        bounds_reg =  []
        best_nlines = 0
        if len(bounding_i) != 0:
            sat_regions = True
            if verbose:
                print('Region %d has %d saturated area(s) at pixels:'%(ireg, len(regions_i_sat)), regions_i_sat)
            for ireg_sat in range(len(regions_l_sat)):
                l_reg_sat = l_reg[regions_i_sat[ireg_sat, 0] : regions_i_sat[ireg_sat, 1]]
                f_reg_sat = f_reg[regions_i_sat[ireg_sat, 0] : regions_i_sat[ireg_sat, 1]]
                n_reg_sat = n_reg[regions_i_sat[ireg_sat, 0] : regions_i_sat[ireg_sat, 1]]
                
                #plt.plot(l_reg_sat,f_reg_sat)
                l_reg_bound_left =  l_reg[bounding_i[ireg_sat, 0][0] : bounding_i[ireg_sat, 0][1]]
                l_reg_bound_right =  l_reg[bounding_i[ireg_sat, 1][0] : bounding_i[ireg_sat, 1][1]]
                f_reg_bound_left =  f_reg[bounding_i[ireg_sat, 0][0] : bounding_i[ireg_sat, 0][1]]
                f_reg_bound_right =  f_reg[bounding_i[ireg_sat, 1][0] : bounding_i[ireg_sat, 1][1]]
                n_reg_bound_left = n_reg[bounding_i[ireg_sat, 0][0] : bounding_i[ireg_sat, 0][1]]
                n_reg_bound_right = n_reg[bounding_i[ireg_sat, 1][0] : bounding_i[ireg_sat, 1][1]]
                
                # set up sat region bounds for evaluating chisq
                width = (l_reg_bound_right[0]-l_reg_bound_left[-1])
                l_bounds_sat = [l_reg_bound_left[0], l_reg_bound_right[-1]]
                l_reg_bounds = np.concatenate((l_reg_bound_left,l_reg_bound_right))
                f_reg_bounds = np.concatenate((f_reg_bound_left,f_reg_bound_right))
                n_reg_bounds = np.concatenate((n_reg_bound_left,n_reg_bound_right))

                # set up grid search for best fit
                N_range = np.linspace(start=logN_bounds[0], stop=logN_bounds[1], num=20)
                b_range = np.linspace(start=np.log10(b_bounds[0]), stop=np.log10(b_bounds[1]), num=20)
                b_range = 10**b_range
                i_middle = int((regions_i_sat[ireg_sat, 0] + regions_i_sat[ireg_sat, 1]) / 2)
                middle_guess = l_reg[i_middle]
                bounds = np.array(np.array([logN_bounds]))
                bounds = np.append(bounds, np.array([b_bounds]), axis=0)
                bounds = np.append(bounds, np.array([l_bounds_sat]), axis=0)

                # find best fit
                chisq_best = 1.e20
                for Ncol in N_range:
                    for bpar in b_range:
                        params = np.array([Ncol, bpar, middle_guess])
                        chisq_soln = _chisq(params, l_reg_bounds, f_reg_bounds, n_reg_bounds, mode)
                        if chisq_soln < chisq_best:
                            chisq_best = chisq_soln
                            Nbest = Ncol
                            bbest = bpar
                params = np.array([Nbest, bbest, middle_guess])
                if verbose:
                    print("Found best-fit sat line (chisq=%g) with params"%(chisq_best), params)


                '''
                for Ncol in [14,15,16,17,18,19,20,21,22,23]:
                    p_guess = Ncol
                    params = np.array([(p_guess),width,middle_guess])
                    bounds = np.array([[12,25]])
                    bounds = np.append(bounds, np.array([b_bounds]), axis=0)
                    bounds = np.append(bounds, np.array([l_bounds_sat]), axis=0)
                    n_lines = int(len(params) / 3)
                    chisq_fcn = lambda *args: _chisq(*args)

                    l_reg_bounds = np.concatenate((l_reg_bound_left,l_reg_bound_right))
                    f_reg_bounds = np.concatenate((f_reg_bound_left,f_reg_bound_right))
                    n_reg_bounds = np.concatenate((n_reg_bound_left,n_reg_bound_right))
                    #plt.plot(l_reg_bounds,f_reg_bounds)

                    soln = minimize(
                        chisq_fcn,
                        params,
                        bounds=bounds,
                        args=(l_reg_bounds,f_reg_bounds, n_reg_bounds, mode),
                        options={"maxiter": 100},
                    )
                    
                    params = soln.x  # set params to new chisq-minimized values
                    chisq_soln = _chisq(params, l_reg_bounds, f_reg_bounds, n_reg_bounds, mode)
                    
                    if chisq_soln < chisq_old:
                        chisq_old = chisq_soln
                        best_nlines += n_lines
                    if verbose:
                        print( "Saturated region %d: %d lines gives chisq=%g (%g) after %d iters"
                            % (ireg_sat, n_lines, chisq_soln, chisq_accept, soln.nit))
                    if chisq_soln < 20:
                        break
                    #plt.plot(l_reg_sat,f_reg_sat)

                params += 0.02 * (
                    2 * np.random.rand(len(params)) - 1
                )  # jiggle params and refit to compute hessian
                soln = minimize(
                    chisq_fcn,
                    params,
                    args=(l_reg_bounds, f_reg_bounds, n_reg_bounds, mode),
                    method="BFGS",
                    options={"maxiter": 100},
                )
                cov = soln.hess_inv  # covariance matrix of final soluiton
                '''
                
                if len(params_reg) == 0:
                    params_reg = np.array(params)
                else:
                    params_reg = np.append(params_reg,np.array(params))
                
                if len(bounds_reg) == 0:
                    bounds_reg = np.array(bounds)
                else:
                    bounds_reg = np.append(bounds_reg, np.array(bounds))
            
            
            
            ## re-compute the parameters for the full region with all saturated lines
            '''
            params_reg += 0.02 * (
                2 * np.random.rand(len(params_reg)) - 1
            )  # jiggle params and refit to compute hessian
            soln = minimize(
                chisq_fcn,
                params_reg,
                args=(l_reg, f_reg, n_reg, mode),
                method="BFGS",
                options={"maxiter": 100},
            )
            chisq_soln = _chisq(params_reg, l_reg, f_reg, n_reg, mode)
            cov = soln.hess_inv  # covariance matrix of final soluiton
            # append lines in this region onto line list
            '''
            if verbose:
                print(
                    "Saturated line gives full region %d (%g-%g): chisq= %g with %d lines"
                    % (
                        ireg,
                        regions_l[ireg, 0],
                        regions_l[ireg, 1],
                        chisq_soln,
                        int(len(params_reg) / 3),
                    )
                )


            params = np.reshape(params_reg,(int(len(params_reg) / 3),3))
            #print(bounds_reg)
            bounds =  np.reshape(bounds_reg,(int(len(params_reg)),2))
            #####Recheck, arbitrary value
            if chisq_soln > 10000:
                params = []
                bounds = []
                sat_regions = False
                print('ChiSquare is too big, probably no saturated region.')

        else:  # fit unsaturated region
            params = []
            bounds = []
            n_lines = 0
            best_nlines = 1
            chisq_old = 1.0e20
            chisq_soln = chisq_old
            chisq_accept = abs(chisq_lim)
            l_reg = l[regions_i[ireg, 0] : regions_i[ireg, 1]]
            f_reg = flux[regions_i[ireg, 0] : regions_i[ireg, 1]]
            n_reg = noise[regions_i[ireg, 0] : regions_i[ireg, 1]]
            

        if len(params) != 0:
            resid = ( 1.0 + f_reg - _tau_to_flux(model_tau(line, params.flatten(), l_reg, mode)) )  # residual spectrum
        else:
            distance = int(len(l_reg)/20)
            if distance < 1:
                distance = 1
            #index,height,widths = find_peaks(f_reg,l_reg,0.2,distance)
            #n_lines = len(index)

        # Look for the absorption peaks in the region.  
        '''
        smoothed = scipy.ndimage.gaussian_filter1d(f_reg, 1.0)  # smooth a little bit to avoid noise peaks
        index,height = scipy.signal.find_peaks(1.-f_reg, prominence=3.*n_reg, height=(3.*np.min(n_reg),1.e20))
        for i_line in index:
            params, bounds = _add_line(params, bounds, l_reg, f_reg, n_reg, float(l0.split()[0]), mode, i_line=i_line)
            n_lines = int(len(params) / 3)
            chisq_soln = _chisq(params, l_reg, f_reg, n_reg, mode)
            if verbose:
                print(f'Region {ireg}: Added peak line {n_lines-1} with N=%.4f, b=%.4f, l=%.4f, chisq=%.3f' % (params[-3],params[-2],params[-1],chisq_soln))
        '''

        # populate a first-guess set of lines in region 
        n_lines = 0
        chisq_best = 1.e20
        chisq_old = 1.e20
        delta_l = l_reg[1]-l_reg[0]
        while n_lines < max_lines-1:
            params, bounds = _add_line(params, bounds, l_reg, f_reg, n_reg, float(l0.split()[0]), mode)
            #if params[-1] in params[2::3]:
            #    print(f'jiggling line {params[-1]} {params[2::3]}')
            #    params[-1] = params[-1] + delta_l * (0.5*np.random.rand() - 1)  
            n_lines = int(len(params) / 3)
            resid = 1.0 + f_reg - _tau_to_flux(model_tau(line, params.flatten(), l_reg, mode))  # residual spectrum
            chisq_soln = _chisq(params, l_reg, f_reg, n_reg, mode)
            if chisq_soln < chisq_best:
                best_nlines = n_lines
                best_params = params
                best_bounds = bounds
                chisq_best = chisq_soln
            if verbose:
                print(f'Region {ireg}: Added line {n_lines-1} with N=%.4f, b=%.4f, l=%.4f, chisq=%.3f' % (params[-3],params[-2],params[-1],chisq_soln))
            if chisq_soln < chisq_accept:  # we're all good :)
                break
            if chisq_soln > 0.99 * chisq_old:  # we're not improving fast enough :(
                break
            if params[-3] == logN_bounds[0] and params[-2] == b_bounds[0]:  # line added isn't significant
                params = np.delete(params, [-3, -2, -1], axis=0)
                bounds = np.delete(bounds, [-3, -2, -1], axis=0)
                n_lines = int(len(params) / 3)
                chisq_soln = _chisq(params, l_reg, f_reg, n_reg, mode)
                break
            chisq_old = chisq_soln
        if verbose:
            print(f'Region {ireg}: Found {n_lines} lines in first guess, chisq=%.3f'%chisq_soln)

        # loop to add lines until desired chisq achieved
        chisq_old = chisq_soln
        while n_lines < max_lines and chisq_soln > chisq_accept:
            params, bounds = _add_line(params, bounds, l_reg, f_reg, n_reg, float(l0.split()[0]), mode)
            n_lines = int(len(params) / 3)
            chisq_fcn = lambda *args: _chisq(*args)
            soln = minimize(
                chisq_fcn,
                params,
                bounds=bounds,
                args=(l_reg, f_reg, n_reg, mode),
                options={"maxiter": _maxiter(n_lines, max_lines)},
            )
            params = soln.x  # set params to new chisq-minimized values
            chisq_soln = _chisq(params, l_reg, f_reg, n_reg, mode)
            if verbose:
                print( "Region %d: Added line %d, chisq=%.3f after %d iters" % (ireg, n_lines, chisq_soln, soln.nit))
            if chisq_soln < chisq_best:
                best_nlines = n_lines
                best_params = params
                best_bounds = bounds
                chisq_best = chisq_soln
            if chisq_soln < 0.95 * chisq_old:
                chisq_old = chisq_soln
            else:
                chisq_old = chisq_soln
                # if after 4 lines fit is not improving (enough), no point in continuing
                if n_lines > 4: 
                    break
            if chisq_soln < chisq_accept:
                break

        if chisq_soln > chisq_best:  # revert to best solution, refit
            params = best_params
            bounds = best_bounds
            n_lines = int(len(params) / 3)
            chisq_soln = _chisq(params, l_reg, f_reg, n_reg, mode)
            if verbose:
                print( "Region %d: Reverting to best with %d lines and chisq=%.3f" % (ireg, n_lines, chisq_soln))
            chisq_fcn = lambda *args: _chisq(*args)
            soln = minimize(
                chisq_fcn,
                params,
                bounds=bounds,
                args=(l_reg, f_reg, n_reg, mode),
                options={"maxiter": _maxiter(n_lines, max_lines)},
            )
            params = soln.x  # set params to new chisq-minimized values
            chisq_soln = _chisq(params, l_reg, f_reg, n_reg, mode)

        '''
        # if any single line strongly exceeds flux decrement anywhere, try removing 
        if chisq_soln > 3*chisq_accept:  
            floor = f_reg - 4. * n_reg
            ip = 0
            while ip < n_lines:
                p_line = [params[3*ip], params[3*ip+1], params[3*ip+2]]
                diff = _tau_to_flux(model_tau(line, p_line, l_reg, mode)) - floor
                if np.any(diff < 0):
                    # remove the line
                    trial_params = np.delete(params, [3*ip, 3*ip+1, 3*ip+2], axis=0)
                    chisq_trial = _chisq(trial_params, l_reg, f_reg, n_reg, mode)
                    if chisq_trial < chisq_best:
                        n_lines = int(len(params) / 3)
                        best_nlines = n_lines
                        best_params = params
                        best_bounds = bounds
                        chisq_best = chisq_soln
                        if verbose:
                            print(f'Region {ireg}: Removed line, chisq improved to {chisq_best}')
                    # try replacing with another line
                    params, bounds = _add_line(params, bounds, l_reg, f_reg, n_reg, float(l0.split()[0]), mode)
                    chisq_fcn = lambda *args: _chisq(*args)
                    soln = minimize(
                        chisq_fcn,
                        params,
                        bounds=bounds,
                        args=(l_reg, f_reg, n_reg, mode),
                        options={"maxiter": _maxiter(n_lines, max_lines)},
                    )
                    params = soln.x  # set params to new chisq-minimized values
                    chisq_trial = _chisq(params, l_reg, f_reg, n_reg, mode)
                    if chisq_trial < chisq_best:
                        n_lines = int(len(params) / 3)
                        best_nlines = n_lines
                        best_params = params
                        best_bounds = bounds
                        chisq_best = chisq_soln
                        if verbose:
                            print(f'Region {ireg}: Replaced line, chisq improved to {chisq_best}')
                ip += 1
        '''

        # jiggle params and refit to compute hessian
        params += 0.02 * ( 2 * np.random.rand(len(params)) - 1)  
        chisq_fcn = lambda *args: _chisq(*args)
        soln = minimize(
            chisq_fcn,
            params,
            args=(l_reg, f_reg, n_reg, mode),
            method="BFGS",
            options={"maxiter": 100},
        )
        cov = soln.hess_inv  # covariance matrix of final soluiton
        chisq_new = _chisq(params, l_reg, f_reg, n_reg, mode)
        if chisq_new < chisq_best:
            if verbose:
                print(f'Region {ireg}: Fit improved with jiggled params chisq={chisq_new}')
            params = soln.x
            n_lines = int(len(params) / 3)
            best_nlines = n_lines
            best_params = params
            best_bounds = bounds
            chisq_best = chisq_new

        # remove small lines as long as chisq doesn't go up by much
        while n_lines > 1:
            trial_params = params
            N_array = np.array([params[ip*3] for ip in np.arange(n_lines)])
            i_del = np.argmin(N_array)
            trial_params = np.delete(params, [i_del, i_del+1, i_del+2], axis=0)
            chisq_trial = _chisq(trial_params, l_reg, f_reg, n_reg, mode)
            delta_chisq = abs(chisq_trial-chisq_soln)/chisq_trial
            if delta_chisq < 0.01 or chisq_trial < chisq_accept:
                if verbose:
                    print("Removing line %d (N=%g)?"%(i_del, N_array[i_del]), chisq_trial, chisq_soln, abs(chisq_trial-chisq_soln)/chisq_trial)
                params = trial_params
                bounds = np.delete(bounds, [i_del, i_del+1, i_del+2], axis=0)
                chisq_soln = chisq_trial
                n_lines = int(len(params)/3)
            else:
                break

        if verbose:
            print(f"Region {ireg}: FINAL FIT {n_lines} lines, N={params[0::3]}, chisq=%.3f"%chisq_soln)
            if chisq_soln > chisq_accept:
                print( "Region %d: WARNING large chisq=%.3f > %.3f; check fit" % (ireg, chisq_soln, chisq_accept))

        # load final line list
        chisq_soln = _chisq(params, l_reg, f_reg, n_reg, mode)
        for ip in np.arange(n_lines):
            line_list["region"] = np.append(line_list["region"], ireg)
            line_list["N"] = np.append(line_list["N"], params[ip * 3])
            line_list["b"] = np.append(line_list["b"], params[ip * 3 + 1])
            line_list["l"] = np.append(line_list["l"], params[ip * 3 + 2])
            line_list["dN"] = np.append(line_list["dN"], np.sqrt(cov[ip * 3, ip * 3]))
            line_list["db"] = np.append(
                line_list["db"], np.sqrt(cov[ip * 3 + 1, ip * 3 + 1])
            )
            line_list["dl"] = np.append(
                line_list["dl"], np.sqrt(cov[ip * 3 + 2, ip * 3 + 2])
            )
            tau_line = model_tau(
                line,
                [params[ip * 3], params[ip * 3 + 1], params[ip * 3 + 2]],
                l_reg,
                mode,
            )
            line_list["EW"] = np.append(
                line_list["EW"], EquivalentWidth(_tau_to_flux(tau_line), l_reg)
            )
            line_list["Chisq"] = np.append(line_list["Chisq"], chisq_soln)
            #if verbose:
            #    print('Region %d: line'%ireg,ip,params[ip*3],params[ip*3+1],params[ip*3+2])#
             # Add temperature for this line
            # Interpolate between the two closest values in the temperature array

            lam = float(line_list['l'][-1]) # take the last line  
            i = np.searchsorted(wavelengths, lam)

            if i == 0: # if before first, take first temperature
                t = temperatures[0]
            elif i >= len(wavelengths): # if after last, take last temperature
                t = temperatures[-1]
            else:
                lam0 = wavelengths[i-1] 
                lam1 = wavelengths[i]
                
                if lam1 == lam0:
                    t = temperatures[i-1]
                else:
                    t = temperatures[i-1] + ((lam - lam0)/(lam1 - lam0) * (temperatures[i]-temperatures[i-1]))

            line_list["t"] = np.append(line_list["t"], t)

            if verbose:
                print('Line at %.3f A: T=%.1f K'%(line_list["l"][-1], line_list["t"][-1]))
            """
            # No interpolation
            idx = _find_nearest(wavelengths, line_list["l"][-1])  # get the last wavelength we just added
            line_list["t"] = np.append(line_list["t"], temperatures[idx])
            print('Line at %.3f A: T=%.1f K'%(line_list["l"][-1], line_list["t"][-1]))
            """

            #if verbose:
            #    print('Region %d: line'%ireg,ip,params[ip*3],params[ip*3+1],params[ip*3+2])



    return line_list



def find_regions(
    wavelengths, fluxes, noise, min_region_width=2, N_sigma=10.0, extend=False, buffer=3, det_flag=False
):
    """
    Finds detection regions above some detection threshold and minimum width.

    Args:
        wavelengths (numpy array)
        fluxes (numpy array): flux values at each wavelength
        noise (numpy array): noise value at each wavelength
        min_region_width (int): minimum width of a detection region (pixels)
        N_sigma (float): detection threshold (std deviations)
        extend (boolean): default is False. Option to extend detected regions untill tau
                        returns to continuum.

    Returns:
        regions_l (numpy array): contains subarrays with start and end wavelengths
        regions_i (numpy array): contains subarrays with start and end indices
    """

    num_pixels = len(wavelengths)
    pixels = range(num_pixels)
    min_pix = 1
    max_pix = num_pixels - 1

    flux_ews = [0.0] * num_pixels
    noise_ews = [0.0] * num_pixels
    det_ratio = [-float("inf")] * num_pixels

    # flux_ews has units of wavelength since flux is normalised. so we can use it for optical depth space
    for i in range(min_pix, max_pix):
        flux_dec = 1.0 - fluxes[i]
        if flux_dec < noise[i]:
            flux_dec = 0.0
        flux_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * flux_dec
        noise_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * noise[i]

    # dev: no need to set end values = 0. since loop does not set end values
    flux_ews[0] = 0.0
    noise_ews[0] = 0.0

    # Range of standard deviations for Gaussian convolution
    std_min = 2
    std_max = 11

    # Convolve varying-width Gaussians with equivalent width of flux and noise
    xarr = np.array([p - (num_pixels - 1) / 2.0 for p in range(num_pixels)])

    # this part can remain the same, since it uses EW in wavelength units, not flux
    for std in range(std_min, std_max):

        gaussian = np.exp(-0.5 * (xarr / std) ** 2)

        flux_func = np.convolve(flux_ews, gaussian, "same")
        noise_func = np.convolve(np.square(noise_ews), np.square(gaussian), "same")

        # Select highest detection ratio of the Gaussians
        for i in range(min_pix, max_pix):
            noise_func[i] = 1.0 / np.sqrt(noise_func[i])
            if flux_func[i] * noise_func[i] > det_ratio[i]:
                det_ratio[i] = flux_func[i] * noise_func[i]

    if det_flag:
        return [], [], det_ratio

    # Select regions based on detection ratio at each point, combining nearby regions
    start = 0
    region_endpoints = []
    for i in range(num_pixels):
        if start == 0 and det_ratio[i] > 0 and fluxes[i] < 1.0:  ##greater 0
            start = i
        elif start != 0 and (det_ratio[i] < 0 or fluxes[i] > 1.0):
            #if (i - start) > min_region_width:
            end = i
            
            
            region_endpoints.append([start, end])
            start = 0

    significant_region_endpoints = []
    for reg in region_endpoints:    
       
        det_ratio = np.array(det_ratio)
        significance = np.sqrt(np.sum(det_ratio[reg[0]:reg[1]]**2))
       # if reg[1]-reg[0] >10:
            #print(reg)
            #print(significance)
            #plt.plot(wavelengths[reg],fluxes[reg])

        if significance == np.inf:
            significance = 0
        if significance > N_sigma: # and reg[1]>60 and reg[0]< (len(fluxes)-60):
            significant_region_endpoints.append(reg)
    # made extend a kwarg option
    # lines may not go down to 0 again before next line starts

    if extend:
        # Expand edges of region until flux goes above 1
        regions_expanded = []
        for reg in significant_region_endpoints:
            start = reg[0]
            i = start
            while i > 0 and fluxes[i] < 1.0:
                i -= 1
            start_new = i
            end = reg[1]
            j = end
            while j < (len(fluxes) - 1) and fluxes[j] < 1.0:
                j += 1
            end_new = j
            regions_expanded.append([start_new, end_new])

    else:
        regions_expanded = significant_region_endpoints
    #print(significant_region_endpoints)
    # Change to return the region indices
    # Combine overlapping regions, check for detection based on noise value
    # and extend each region again by a buffer
    regions_l = []
    regions_i = []
    buffer = buffer
    for i in range(len(regions_expanded)-1):
        #print(len(regions_expanded),i)
        if len(regions_expanded) == i:
            break
        start = regions_expanded[i][0]
        end = regions_expanded[i][1]
        #print(start,end)
        #print(regions_expanded[i+1])
        #print('difference'+str(regions_expanded[i+1][0]-end))
        if len(regions_expanded) == i+1:
            break
        if (regions_expanded[i+1][0]-end) < 5:
            regions_expanded[i][1] = regions_expanded[i+1][1]
            regions_expanded=np.delete(regions_expanded,(i+1),axis=0)
        #print(regions_expanded)
        
        end_init = end
    for i in range(len(regions_expanded)):
        
        start = regions_expanded[i][0]
        end = regions_expanded[i][1]
        # TODO: this part seems to merge regions if they overlap - try printing this out to see if it can be modified to not merge regions?
        #print((len(regions_expanded) - 1), regions_expanded[i + 1][0],end)
        #if i < (len(regions_expanded) - 1) and end > regions_expanded[i + 1][0]:
        #    end = regions_expanded[i + 1][1]
        #    print('merged')
            
        for j in range(start, end):
            
            flux_dec = 1.0 - fluxes[j]
            #if flux_dec > abs(noise[j]):# * N_sigma:
            if start >= buffer:
                start -= buffer
            if end < len(wavelengths) - buffer:
                end += buffer
            regions_l.append([wavelengths[start], wavelengths[end]])
            regions_i.append([start, end])
            
            break
    
    if environment.verbose >= environment.VERBOSE_TACITURN:
        print("Found {} detection regions".format(len(regions_l)))
    return np.array(regions_l), np.array(regions_i), det_ratio




def find_regions_sat(
    wavelengths, fluxes, noise, min_region_width=2, N_sigma=10.0, extend=False
):
    """
    Finds detection regions above some detection threshold and minimum width.

    Args:
        wavelengths (numpy array)
        fluxes (numpy array): flux values at each wavelength
        noise (numpy array): noise value at each wavelength
        min_region_width (int): minimum width of a detection region (pixels)
        N_sigma (float): detection threshold (std deviations)
        extend (boolean): default is False. Option to extend detected regions untill tau
                        returns to continuum.

    Returns:
        regions_l (numpy array): contains subarrays with start and end wavelengths
        regions_i (numpy array): contains subarrays with start and end indices
    """

    num_pixels = len(wavelengths)
    pixels = range(num_pixels)
    min_pix = 1
    max_pix = num_pixels - 1

    flux_ews = [0.0] * num_pixels
    noise_ews = [0.0] * num_pixels
    det_ratio = [-float("inf")] * num_pixels

    # flux_ews has units of wavelength since flux is normalised. so we can use it for optical depth space
    for i in range(min_pix, max_pix):
        flux_dec = 1.0 - fluxes[i]
        if flux_dec < noise[i]:
            flux_dec = 0.0
        flux_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * flux_dec
        noise_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * noise[i]

    # dev: no need to set end values = 0. since loop does not set end values
    flux_ews[0] = 0.0
    noise_ews[0] = 0.0

    # Range of standard deviations for Gaussian convolution
    std_min = 2
    std_max = 11

    # Convolve varying-width Gaussians with equivalent width of flux and noise
    xarr = np.array([p - (num_pixels - 1) / 2.0 for p in range(num_pixels)])

    # this part can remain the same, since it uses EW in wavelength units, not flux
    for std in range(std_min, std_max):

        gaussian = np.exp(-0.5 * (xarr / std) ** 2)

        flux_func = np.convolve(flux_ews, gaussian, "same")
        noise_func = np.convolve(np.square(noise_ews), np.square(gaussian), "same")

        # Select highest detection ratio of the Gaussians
        for i in range(min_pix, max_pix):
            noise_func[i] = 1.0 / np.sqrt(noise_func[i])
            if flux_func[i] * noise_func[i] > det_ratio[i]:
                det_ratio[i] = flux_func[i] * noise_func[i]

    # Select regions based on detection ratio at each point, combining nearby regions
    start = 0
    region_endpoints = []
    index = np.where(np.abs(det_ratio-np.max(det_ratio)) < N_sigma)[0]
    
    for i in range(num_pixels):
        if start == 0 and np.abs(det_ratio[i]-np.max(det_ratio)) < N_sigma: # and fluxes[i] < 1.0:
            start = i
        elif start != 0 and  np.abs(det_ratio[i]-np.max(det_ratio)) > N_sigma: # or fluxes[i] > 1.0):
            if (i - start) > min_region_width:
                end = i
                region_endpoints.append([start, end])
            start = 0

    # made extend a kwarg option
    # lines may not go down to 0 again before next line starts

    if extend:
        # Expand edges of region until flux goes above 1
        regions_expanded = []
        for reg in region_endpoints:
            start = reg[0]
            i = start
            while i > 0 and fluxes[i] < 1.0:
                i -= 1
            start_new = i
            end = reg[1]
            j = end
            while j < (len(fluxes) - 1) and fluxes[j] < 1.0:
                j += 1
            end_new = j
            regions_expanded.append([start_new, end_new])

    else:
        regions_expanded = region_endpoints

    # Change to return the region indices
    # Combine overlapping regions, check for detection based on noise value
    # and extend each region again by a buffer
    regions_l = []
    regions_i = []
    bounding_regions_i = []
    buffer = 3
    for i in range(len(regions_expanded)):
        start = regions_expanded[i][0]
        end = regions_expanded[i][1]
        end_init = end
        # TODO: this part seems to merge regions if they overlap - try printing this out to see if it can be modified to not merge regions?
        if i < (len(regions_expanded) - 1) and np.abs(end -regions_expanded[i + 1][0])<2:
            end = regions_expanded[i + 1][1]
        for j in range(start, end):
            flux_dec = 1.0 - fluxes[j]
            if flux_dec > abs(noise[j]) * N_sigma:
                if start >= buffer:
                    start -= buffer
                if end < len(wavelengths) - buffer:
                    end += buffer
                regions_l.append([wavelengths[start], wavelengths[end]])
                regions_i.append([start, end])

                if (start-18)<0:
                    start1 = 0
                else:
                    start1 = (start-18)
                if (start-8)< 0:
                    start2 = start
                else:
                    start2 = (start-8)
                if (end+8) >len(fluxes):
                    end1 = end
                else:
                    end1 = (end+8)
                if (end+18) > len(fluxes):
                    end2 = int(len(fluxes))
                else:
                    end2 = (end+18)
                
                bounding_regions_i.append([[start1,start2],[end1,end2]])

                #print(regions_l,regions_i)
                break

    if environment.verbose >= environment.VERBOSE_TACITURN:
        print("Found {} saturated regions".format(len(regions_l)))

    #bounding_regions_i = [regions_i[0]-10,regions_i[-1]+10]
    return np.array(regions_l), np.array(regions_i), np.array(bounding_regions_i)





def model_tau(line, p, l, mode="Voigt"):
    """
    Compute optical depth vs. wavelength for a set of lines.

    Args:
        p (numpy array): [logN,b,wavelength] for a set of lines, in
                         a flattened numpy array.
        l (numpy array): Wavelengths over which to compute model spectrum.
    Returns:
        total_tau:  Optical depths (vs. l) from the combined set of lines
    """
    p = np.array(p)
    total_tau = np.zeros(len(l), dtype=float)
    if len(p) == 0:
        return total_tau  # no lines yet, return zeros
    for ip in range(int(len(p) / 3)):
        _, tau = line_profile(
            line, 10 ** p[ip * 3], b=p[ip * 3 + 1], l0=p[ip * 3 + 2], l=l, mode=mode
        )
        total_tau += tau  # add up optical depths from all lines
    return total_tau


def EquivalentWidth(fluxes, waves):
    """
    Find the equivalent width of a line/region.

    Args:
        taus (numpy array): the optical depths.
        waves (numpy array): list of wavelength for region.
    Returns:
        Equivalent width in units of waves.
    """
    dEW = np.zeros(len(fluxes))
    for i in range(1, len(fluxes) - 1):
        dEW[i] = (1.0 - fluxes[i]) * abs(waves[i + 1] - waves[i - 1]) * 0.5
    dEW[0] = (1.0 - fluxes[0]) * abs(waves[1] - waves[0])
    dEW[i - 1] = (1.0 - fluxes[i - 1]) * abs(waves[i - 1] - waves[i - 2])
    return np.sum(dEW)



# Main to run as standalone code 

if __name__ == '__main__':

    spec_file = sys.argv[1]
    if len(sys.argv) > 2:
        vel_range = float(sys.argv[2])
    else:
        vel_range = 600.

    #spec_file = 'sample_galaxy_811_MgII2796_270_deg_0.75r200.h5'
    
    #if 'SiIII1206' in spec_file.split('_'):
    print('Doing spectrum file',spec_file)
    spec = Spectrum(f'{spec_file}')
    spec.main(vel_range=vel_range, chisq_unacceptable=25, chisq_asym_thresh=-3.0, write_lines=True, plot_fit=True)
    
