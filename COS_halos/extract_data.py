# extract_cos_halos.py
# Extracts OVI measurements from COS-Halos via pyigm and saves to HDF5
# for comparison with SIMBA-C mock spectra results.
#
# Output HDF5 contains per-galaxy:
#   - OVI equivalent width (rest-frame, Angstrom)
#   - OVI column density log N (cm^-2)
#   - Detection flag (1=detection, 0=upper limit)
#   - Impact parameter rho (kpc, physical)
#   - Stellar mass log M* (log Msun)
#   - log sSFR (log yr^-1), -14.0 for quenched
#   - Redshift
#   - SF/GV/Q category
#   - cgm_id string

import numpy as np
import h5py
from pyigm.cgm.cos_halos import COSHalos

line = 'OVI 1031'   # pyigm line label
ion = 'OVI' # pyigm ion label
wavelength = '1031.9261' # pyigm ion wavelength

EW_UPPER_LIMIT_FLAG = 0
EW_DETECTION_FLAG   = 1


def quench_thresh(z):
    """
    Quenching threshold in log sSFR [yr^-1]
    """
    return -1.8 + 0.3 * z - 9.0


def ssfr_category(log_ssfr, z):
    """
    Define the SF/GV/Q thresholds
    """
    thresh = quench_thresh(z)
    if log_ssfr == -14.0:
        return 'Q'
    elif log_ssfr >= thresh:
        return 'SF'
    elif log_ssfr >= thresh - 1.0:
        return 'GV'
    else:
        return 'Q'

def get_impact_parameter(cgm):
    """
    Return impact parameter in physical kpc.
    COS-Halos stores rho on cgm.rho (an astropy Quantity).
    """
    rho_kpc = float(cgm.rho.to('kpc').value)
    return rho_kpc


def get_galaxy_properties(cgm):
    """
    Return (log_mstar, log_ssfr, redshift) from the galaxy object.
    Returns (nan, -14.0, nan) on failure.
    """
    gal = cgm.galaxy

    # Stellar mass in log Msun in COS-Halos pyigm objects
    try:
        log_mstar = float(gal.stellar_mass)
    except Exception:
        return np.nan, -14.0, np.nan

    # Redshift
    try:
        z = float(gal.z)
    except Exception:
        return np.nan, -14.0, np.nan

    # Convert SFR to sSFR
    try:
        sfr_data = gal.sfr                 # tuple: ('yes'/'no', value, indicator)
        sfr      = float(sfr_data[1])
        mstar_lin = 10.0 ** log_mstar      # convert log to linear for division
        if sfr > 0 and mstar_lin > 0:
            log_ssfr = float(np.log10(sfr / mstar_lin))
        else:
            log_ssfr = -14.0
    except Exception:
        log_ssfr = -14.0

    # Virial radius

    try:
        r200 = float(gal.rvir)  # in kpc/h
        print("Virial radius found!")
    except Exception:
        r200 = np.nan

    return log_mstar, log_ssfr, z, r200


def extract_cos_halos(out_file):
    """
    Load COS-Halos, extract all relevant quantities, and save to HDF5.
    """

    print('Loading COS-Halos...')
    survey = COSHalos()
    survey.load_sys()
    print(f'  Loaded {len(survey.cgm_abs)} CGM systems')

    # Get EW and column density table for OVI 1031 in one shot
    ovi_tbl = survey.trans_tbl('OVI 1031')

    records = []

    for i, cgm in enumerate(survey.cgm_abs):
        cgm_id = str(cgm.name)

        log_mstar, log_ssfr, z, r200 = get_galaxy_properties(cgm)
        if np.isnan(log_mstar) or np.isnan(z):
            print(f'  [SKIP] {cgm_id}: missing mass or redshift')
            continue

        rho_kpc = get_impact_parameter(cgm)
        cat = ssfr_category(log_ssfr, z)

        ew      = ovi_tbl['EW'][i]
        ew_err  = ovi_tbl['sig_EW'][i]
        log_N   = ovi_tbl['logN'][i]
        det_flag = EW_DETECTION_FLAG if ovi_tbl['flag_EW'][i] == 1 else EW_UPPER_LIMIT_FLAG

        records.append(dict(
            cgm_id    = cgm_id,
            log_mstar = log_mstar,
            log_ssfr  = log_ssfr,
            redshift  = z,
            rho_kpc   = rho_kpc,
            r200      = r200,
            ew        = ew,
            ew_err    = ew_err,
            category  = cat,
            det_flag  = det_flag,
            log_N     = log_N,
        ))

    if not records:
        raise RuntimeError('No records extracted from COS-Halos.')

    cats = [r['category'] for r in records]
    print(f'\nExtracted {len(records)} galaxies  '
          f'({cats.count("SF")} SF, {cats.count("GV")} GV, {cats.count("Q")} Q)')
    dets = sum(r['det_flag'] == 1 for r in records)
    print(f'OVI detections: {dets}/{len(records)}')

    with h5py.File(out_file, 'w') as hf:
        hf.create_dataset('cgm_id',    data=np.array([r['cgm_id']    for r in records], dtype='S40'))
        hf.create_dataset('log_mstar', data=np.array([r['log_mstar'] for r in records]))
        hf.create_dataset('log_ssfr',  data=np.array([r['log_ssfr']  for r in records]))
        hf.create_dataset('redshift',  data=np.array([r['redshift']  for r in records]))
        hf.create_dataset('rho_kpc',   data=np.array([r['rho_kpc']   for r in records]))
        hf.create_dataset('r200',   data=np.array([r['r200']   for r in records]))
        hf.create_dataset('ew_AA',     data=np.array([r['ew']        for r in records]))
        hf.create_dataset('ew_err_AA',     data=np.array([r['ew_err']        for r in records]))
        hf.create_dataset('det_flag',  data=np.array([r['det_flag']  for r in records], dtype=int))
        hf.create_dataset('log_N',     data=np.array([r['log_N']     for r in records]))
        hf.create_dataset('category',  data=np.array([r['category']  for r in records], dtype='S2'))

        hf.attrs['survey']      = 'COS-Halos'
        hf.attrs['line']    = line
        hf.attrs['n_total']     = len(records)
        hf.attrs['n_sf']        = cats.count('SF')
        hf.attrs['n_gv']        = cats.count('GV')
        hf.attrs['n_q']         = cats.count('Q')
        hf.attrs['n_detections']= dets

    print(f'\nSaved to {out_file}')
    return records


if __name__ == '__main__':
    out_file = '/home/matylda/SHP/Data/samples/cos_halos_ovi_1031.h5'
    extract_cos_halos(out_file)