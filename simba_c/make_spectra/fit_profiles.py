"""
Runs the vpfitter for the spectra generated for different
different sightlines.

Requires pipeline.py to be run for the whole galaxy sample / selected galaxies.  
"""


import os
import sys
import numpy as np
from spectrum import Spectrum

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    ion = sys.argv[4]
    azimuth = sys.argv[5]

    vel_range = 600.
    chisq_asym_thresh = -3
    chisq_unacceptable = 25

    spec_dir = f'/disk04/mrejus/sh/normal/{model}_{wind}_{snap}_hm12/'
    listdir = os.listdir(spec_dir)
    spec_file = [i for i in listdir if ion and azimuth in i]
    
    for my_file in spec_file:
        spec = Spectrum(f'{spec_dir}{my_file}')
        print('Fitting lines in: %s' % my_file)

        spec.main(
            vel_range=vel_range,
            do_fit=False,
            write_lines=False,
            chisq_unacceptable=chisq_unacceptable,
            chisq_asym_thresh=chisq_asym_thresh, 
            plot_fit=True,
            )
