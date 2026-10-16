# TODO: Justin will embed this logic into two stages in sompz.py 
# previous one SOM work has used ceci stages for SOM training and assignment, but not for the p(z|c) binning and stacking done her
# TODO: SOMPZnz_onesom
# TODO: SOMPZTomobin_onesom

import os

import yaml
import numpy as np
import qp
import tables_io

import matplotlib.pyplot as plt

datadir = './'

with open(os.path.join(os.path.dirname(__file__), 'cardinal_estimate_config.yml')) as f:
    estimate_config = yaml.safe_load(f)

infile_deep_data = estimate_config['som_deepdeep_estimator']['data']
infile_deep_assignments = os.path.join(datadir, 'assignment_som_deepdeep_estimator.hdf5')
infile_spec_assignments = os.path.join(datadir, 'assignment_som_deepspec_estimator.hdf5')
infile_pz_c = os.path.join(datadir, 'pz_c.hdf5')

pzc_config = estimate_config['som_pzc_stage']
zbins_min = pzc_config['zbins_min']
zbins_max = pzc_config['zbins_max']
zbins_dz = pzc_config['zbins_dz']
zbins = np.arange(zbins_min - zbins_dz / 2., zbins_max + zbins_dz, zbins_dz)
zmids = 0.5 * (zbins[1:] + zbins[:-1])
nbins = 5 # set by SOMPZTomobin_onesom cfg yml

n_cells = np.prod(estimate_config['som_deepdeep_estimator']['som_shape'])

nz = np.zeros((nbins, len(zmids)))

def get_mean(zmids,hists):
    normalization = np.sum(hists)
    if normalization == 0:
        normalization = 1
    return np.sum(zmids*hists)/normalization

def get_cell_weights(deep_assignment, deep_data, n_cells=n_cells):

    # get weight based on cell occupation
    cells, cell_counts = np.unique(deep_assignment, return_counts=True)
    weights = np.zeros(n_cells)
    # weights[cells] = cell_counts[cells]  # old weighting based on cell occupation

    # get weights from column
    for cell in cells:
        sel = (deep_assignment == cell)
        # weights[cell] = np.sum(deep_data['i_hsmshaperegauss_derived_weight'][sel])
        weights[cell] = len(deep_data['redshift'][sel])

    # convert to shape to be multiplied with pz_c
    weights = weights[:, np.newaxis]
    
    return weights

# rank p(z|c) by <z|c>
pz_c = tables_io.read(infile_pz_c)['pz_c']
meanz_c = np.array([get_mean(zmids, pz_c[i]) for i in range(len(pz_c))])
order_by_meanz_c = np.argsort(meanz_c)

# construct bins to yield equal counts of WL sample galaxies
assignments = tables_io.read(infile_deep_assignments)['cells']
data = tables_io.read(infile_deep_data)
weights = get_cell_weights(assignments, data)
cell_counts = weights[:, 0]

ngal = cell_counts.sum()
target_per_bin = ngal / nbins

# iterate over cells in order of increasing <z|c>, split when cumulative count reaches each target
occupied_cells = order_by_meanz_c[cell_counts[order_by_meanz_c] > 0]
cumsum = np.cumsum(cell_counts[occupied_cells])
split_at = [
    np.searchsorted(cumsum, (b + 1) * target_per_bin, side='right')
    for b in range(nbins - 1)
]

cells_by_bin = []
start = 0
for stop in split_at + [len(occupied_cells)]:
    cells_by_bin.append(occupied_cells[start:stop])
    start = stop

# sum cells to construct n(z)
for i in range(nbins):
    cells_i = cells_by_bin[i]
    bin_ngal = cell_counts[cells_i].sum()
    nz[i, :] = np.sum(pz_c[cells_i] * weights[cells_i], axis=0)
    print(
        f"Bin {i}: {len(cells_i)} cells, {bin_ngal:.0f} galaxies "
        f"(target {target_per_bin:.0f}, <z|c> range "
        f"{meanz_c[cells_i].min():.3f}–{meanz_c[cells_i].max():.3f})"
    )

tomo_ens = qp.Ensemble(qp.interp, data=dict(xvals=zmids, yvals=nz))
outfile_nz = os.path.join(datadir, 'NZ_ESTIMATED_SINGLE_SOM.hdf5')
tomo_ens.write_to(outfile_nz)
print(f"Wrote {outfile_nz}")

# plot n(z)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for i in range(nbins):
    ax.plot(zmids, nz[i], label=f'Bin {i+1}')
ax.legend()
ax.set_xlabel('z')
ax.set_ylabel('n(z)')
plt.savefig(os.path.join(datadir, 'NZ_ESTIMATED_SINGLE_SOM.png'))
plt.close()