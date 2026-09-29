import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import h5py
from scipy.optimize import curve_fit
from scipy import interpolate
import os
import sys
sys.path.insert(0, '/home/matylda/sh/make_spectra_sh/')
from utils import *
from physics import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

def power_law(x, a, b):
    return x*a + b

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    type = sys.argv[4]  # 'azimuth', 'chisqion', 'mass'


    #lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    #plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  #r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']

    # IGNORE ABOVE AND FOCUS ON OVI1031 ONLY

    lines = ["OVI1031"]
    plot_lines = [r'${\rm OVI}\ 1031$']
    
    plot_dir = f'/home/matylda/data/plots/'

    ncells = 16
    start = [13.75, 12.75, 13.75, 12.75, 13.75, 13.75]
    end = [15.5, 14.5, 15.5, 14.5, 15.5, 14.5]
    logN = np.arange(9, 18, 0.01)

    cddf_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_OVI1031_cddf_{type}.h5'
    
    plot_data = read_h5_into_dict(cddf_file)
    
    plt.plot(plot_data['plot_logN'], plot_data[f'cddf_all'], c='red', ls='solid', lw=1, label = "Data")
    mask = ~np.isinf(plot_data['cddf_all'])
    #print(mask)
    logN_use = plot_data['plot_logN'][mask]
    data_use = plot_data['cddf_all'][mask]
    print(plot_data['cddf_all'])
    start_i = np.argmax(data_use)
    
    popt, pcov = curve_fit(power_law, logN_use[start_i:], data_use[start_i:])
    power_law_fit = logN*popt[0]+ popt[1]
    plt.plot(logN, power_law_fit, c='tab:pink', lw=1, ls='--', label = "Power-law fit")
    
    mask = ~np.isinf(plot_data['cddf_all']) 
    f = interpolate.interp1d(plot_data['plot_logN'][mask], plot_data['cddf_all'][mask], fill_value='extrapolate')
    
    cddf_extra = f(logN)
    plt.plot(logN, cddf_extra, c='skyblue', lw=1, ls='--', label = "Extrapolation")
    plt.xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')
    plt.ylabel(r'${\rm log }(\delta^2 n / \delta X \delta N )$')
    plt.title("Completeness")
    
    end_i = np.argmax(cddf_extra) + 10
    
    plot_data['completeness'] = logN[np.argmin(np.abs((10**cddf_extra[:end_i] / 10**power_law_fit[:end_i] ) - 0.5))]
    plt.axvline(plot_data['completeness'], c='k', ls='--', lw=1)
    
    write_dict_to_h5(plot_data, cddf_file)
    
    plt.tight_layout()
    plt.legend()
    #fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_cddf_completeness.png')
    #plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_cddf_completeness_extras.png')
    #plt.show()
    plt.close()
