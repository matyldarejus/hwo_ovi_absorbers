# plot_ew_profile.py
# Plot the equivalent width vs impact parameter for OVI absorbers in the extracted COS-Halos galaxies 
# As in Fig 5 of Appleby 2021 -- this is just diagnostic 

import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['savefig.dpi'] = 400

cb_blue = '#5289C7'
cb_red  = '#E26F72'

cos_file = '/home/matylda/SHP/Data/samples/cos_halos_ovi_1031.h5'
with h5py.File(cos_file, 'r') as f:
    rho_kpc  = f['rho_kpc'][:]
    r200     = f['r200'][:]
    ew       = f['ew_AA'][:]
    ew_err   = f['ew_err_AA'][:]
    det_flag = f['det_flag'][:]
    category = np.array([c.decode() for c in f['category'][:]])

rho_fr200 = rho_kpc / r200
log_ew    = np.log10(ew)
log_ew_err = ew_err / (ew * np.log(10))  # propagate error in log

sf_mask = category == 'SF'
q_mask  = category == 'Q'

fig, ax = plt.subplots(figsize=(7, 5))

for mask, color, label in [(sf_mask, cb_blue, 'COS-Halos SF'), (q_mask, cb_red, 'COS-Halos Q')]:
    det  = mask & (det_flag == 1)
    ul   = mask & (det_flag == 0)

    ax.errorbar(rho_fr200[det], log_ew[det], yerr=log_ew_err[det],
                fmt='o', color=color, ms=6, lw=1.2, capsize=3, label=label)
    ax.errorbar(rho_fr200[ul], log_ew[ul], yerr=0.1,
                fmt='v', color=color, ms=6, lw=1.2, capsize=0,
                uplims=True) 

ax.set_xlabel(r'$r_\perp / r_{200}$')
ax.set_ylabel(r'log (EW / $\AA$)')
ax.legend()
plt.tight_layout()
plt.savefig('/home/matylda/SHP/Plots/cos_comparison/cos_halos_ovi_profile.png', dpi=400)
plt.show()