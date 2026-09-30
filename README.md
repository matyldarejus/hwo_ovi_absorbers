# HWO OVI Absorbers
All the code I needed to carry out my senior honours project 'Finding the Missing Baryons with NASA's Next Generation Telescope' supervised by Prof Romeel Dave at the University of Edinburgh. The purpose of the project was to generate and fit mock absorption spectra of OVI lines to simulate the observations taken by the upcoming Habitable Worlds Observatory, and predict the conclusions on baryonic makeup of CGM based on them.   

The main code is located in the `simba_c` folder, consisting of main two parts:

- `make_spectra` in order of how the code should be run:
    - `get_galaxy_sample.py`: Picks out galaxy sample in mass and sSFR/SFR bins.
    - `select_los_particles.py`: Samples particles in lines-of-sight around each galaxy.
    - `save_new_dataset.py`: Saves data for particles only overlapping the lines-of-sight to save memory. 
    - `generate_spectra.py`: Generates mock absorption spectra using Pygad routines.
    - `pipeline.py`: Run `generate_spectra.py` for each galaxy for each line-of-sight.
    - `spectrum.py`: Fit Voigt profiles to absorption around each galaxy
    - `fit_profiles.py`: Run `spectrum.py` routines for each galaxy.
    - `Extras` inc. `get_gal_sm_ssfr.py`, `get_sample_temp.py`, `plot_galaxy_sample.py`, `plot_galaxy_sample_ssfr.py`: Plot galaxy samples and calculate misc properties.

- `analyse_spectra`: calculates CDDF split by mass, azimuth, sSFR, etc. and other misc properties for plotting.

The other folders within the repo are:
- `plots`: Code to create various plots, mostly pretty well annotated throughout to indicate what is being plotted.
- `COS_halos`: WIP code for producing comparisons of generated galaxy sample to the [COS-Halos survey](https://www.stsci.edu/~tumlinso/COS-Halos/Welcome.html).

Environment in which the code was tested out to run best: 
python=3.9 yt=4.2.1 matplotlib=3.4.3 numpy=1.22 scipy=1.9


## Acknowledgements

This repository is based on code originally developed by Sarah Appleby as part of her PhD project under the supervision of Romeel Dave at The University of Edinburgh.

The original version can be found [here](https://github.com/sarahappleby/cgm/tree/master/absorption/ml_project).  

The code was modified to work with `simba-c` snapshots, improved with a new version of Voigt profile fitter, and has new plots and statistics. 


