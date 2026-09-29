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
    #spec_dir = f'./test/'
    listdir = os.listdir(spec_dir)
    spec_file = [i for i in listdir if ion and azimuth in i] # Check how many repeats this does, if I need to run it once or over x number of galaxies
    
    #print(listdir)

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
