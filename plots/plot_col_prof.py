import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.lines import Line2D
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.frameon'] = True
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['figure.dpi'] = 130

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
    cmap_list = cmap(np.linspace(minval, maxval, n))
    cmap_list[:, -1] = alpha
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})", cmap_list)
    return new_cmap


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]


    line = "OVI1031"
    plot_label = r'${\rm OVI}\ 1031$'
    chisq_lim_dict = {'snap_151': 4.5}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']
    N_min = 13.2
    Tphoto_ovi = 5.0  # log(T) threshold for collisional OVI


    linestyle = ':'
    marker = '^'


    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200 + 1) * delta_fr200, delta_fr200)


    delta_m = 0.5
    min_m = 10.
    nbins_m = 3
    mass_bins = np.arange(min_m, min_m + (nbins_m + 1) * delta_m, delta_m)
    mass_plot_titles = [
        f'{mass_bins[i]}' + r'$ < \textrm{log} M_\star < $' + f'{mass_bins[i+1]}'
        for i in range(nbins_m)
    ]

 
    idelta = 1. / (len(mass_bins) - 1)
    icolor = np.arange(0., 1. + idelta, idelta)
    cmap = cm.get_cmap('magma')
    cmap = truncate_colormap(cmap, 0.2, .8)
    mass_colors = [cmap(i) for i in icolor]


    plot_dir = '/home/matylda/data/plots/'
    sample_dir = '/disk04/mrejus/sh/samples/'
    results_file = f'/disk04/mrejus/sh/normal/results/{model}_{wind}_{snap}_hm12_fit_lines_{line}.h5'


    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]

    fig, ax = plt.subplots(1, 1)

    mass_lines = [Line2D([0, 1], [0, 1], color=mass_colors[i], ls=':', marker=marker, lw=1.5)
              for i in range(len(mass_colors))]
    leg1 = ax.legend(mass_lines, mass_plot_titles, loc=3, fontsize=20)
    ax.add_artist(leg1)


    ion_line = [Line2D([0, 1], [0, 1], ls=linestyle,
                       marker=marker, color='dimgrey')]


    fcol = np.zeros((len(mass_plot_titles), len(fr200)))
    ncol = np.zeros((len(mass_plot_titles), len(fr200)))
    ntotal = np.zeros((len(mass_plot_titles), len(fr200)))


    with h5py.File(results_file, 'r') as hf:
        for i, r in enumerate(fr200):

            # Load OVI data
            try:
                all_T = np.array(hf[f'log_T_{r}r200'][:])
                all_N = np.array(hf[f'log_N_{r}r200'][:])
                all_chisq = np.array(hf[f'chisq_{r}r200'][:])
                all_ids = np.array(hf[f'ids_{r}r200'][:])
            except KeyError:
                print(f"No data for r = {r}r200, skipping.")
                continue

            # Apply quality cuts
            mask = (all_N > N_min) & (all_chisq < chisq_lim)
            all_T, all_N, all_ids = all_T[mask], all_N[mask], all_ids[mask]

            if len(all_ids) == 0:
                print(f"No absorbers for r = {r:.2f}r200, skipping.")
                continue

            # Match galaxy IDs → stellar masses
            idx = np.concatenate(
                [np.where(gal_ids == gid)[0] for gid in all_ids if gid in gal_ids],
                axis=0) if len(all_ids) > 0 else np.array([], dtype=int)

            if len(idx) == 0:
                print(f"No matching galaxy IDs for r = {r:.2f}r200, skipping.")
                continue

            all_mass = mass[idx.astype(int)]
            collisional = all_T > Tphoto_ovi

            for j in range(len(mass_plot_titles)):
                mass_mask = (all_mass > mass_bins[j]) & (
                    all_mass < mass_bins[j + 1])
                total_absorption = np.nansum(10**all_N[mass_mask])
                if total_absorption > 0:
                    fcol[j][i] = np.nansum(
                        10**all_N[mass_mask & collisional]) / total_absorption
                else:
                    fcol[j][i] = np.nan

                ncol[j][i] = np.sum(mass_mask & collisional)
                ntotal[j][i] = np.sum(mass_mask)

    for j in range(len(mass_plot_titles)):
        plt.plot(fr200, fcol[j], color=mass_colors[j],
                 ls=linestyle, marker=marker, lw=1.5)

    plt.ylim(-0.05, 1)
    plt.ylabel(r'$\sum N_{\rm col} / \sum N_{\rm total}$')
    plt.xlabel(r'$r_\perp / r_{200}$')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_fcol_OVI.png', format='png')
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_fcol_OVI.pdf', format='pdf')
    plt.close()