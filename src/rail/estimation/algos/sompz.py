"""
Port of SOMPZ
"""
import os
from ceci.config import StageParameter as Param
from rail.core.data import TableHandle, ModelHandle, QPHandle, Hdf5Handle
from rail.estimation.estimator import CatEstimator, CatInformer
import rail.estimation.algos.som as somfuncs
from rail.core.common_params import SHARED_PARAMS
from multiprocessing import Pool
import numpy as np
import qp
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import pickle
import gc
import inspect

class Pickableclassify:  # pragma: no cover
    def __init__(self, som, flux, fluxerr, inds):
        self.som = som
        self.flux = flux
        self.inds = inds
        self.flux_err = fluxerr

    def __call__(self, ind):
        cells_test, dist_test = self.som.classify(self.flux[self.inds[ind]], self.flux_err[self.inds[ind]])
        return cells_test, dist_test


def_bands = ["u", "g", "r", "i", "z", "y"]
default_bin_edges = [0.0, 0.405, 0.665, 0.96, 2.0]
default_input_names = []
default_err_names = []
default_zero_points = []
for band in def_bands:
    default_input_names.append(f"mag_{band}_lsst")
    default_err_names.append(f"mag_err_{band}_lsst")
    default_zero_points.append(30.)

def mag2flux(mag, zero_pt=30):
    # zeropoint: M = 30 <=> f = 1
    exponent = (mag - zero_pt) / (-2.5)
    val = 1 * 10 ** (exponent)
    return val

def magerr2fluxerr(magerr, flux):
    coef = np.log(10) / -2.5
    return np.abs(coef * magerr * flux)

def gaussian_rbf(weight_map, central_index, cND, map_shape, scale_length=1, max_length=0, **kwargs):
    # fills weight map with gaussian kernel exp(-0.5 (distance / scale_length) ** 2)

    w_dims = len(weight_map)
    map_dims = len(map_shape)
    inv_scale_length_square = scale_length ** -2.

    if max_length <= 0:
        max_length_square = np.inf
    else:
        max_length_square = max_length ** 2

    # update all cells
    for c in range(w_dims):
        # convert to ND
        np.unravel_index(c, map_shape, cND)

        # get distance, including accounting for toroidal topology
        diff2 = 0.0
        for di in range(map_dims):
            best_c = central_index[di]
            ci = cND[di]
            i_dims = map_shape[di]
            diff = (ci - best_c)
            while diff < 0:
                diff += i_dims
            while diff >= i_dims * 0.5:
                diff -= i_dims
            diff2 += diff * diff
            if diff2 > max_length_square:
                continue

        if diff2 <= max_length_square:
            weight_map[c] = np.exp(-0.5 * diff2 * inv_scale_length_square)

def calculate_pcchat(deep_som_size, wide_som_size, cell_deep_assign, cell_wide_assign, overlap_weight):
    pcchat_num = np.zeros((deep_som_size, wide_som_size))
    np.add.at(pcchat_num,
              (cell_deep_assign, cell_wide_assign),
              overlap_weight)

    pcchat_denom = pcchat_num.sum(axis=0)
    pcchat = pcchat_num / pcchat_denom[None]

    # any nonfinite in pcchat are to be treated as 0 probabilty
    pcchat = np.where(np.isfinite(pcchat), pcchat, 0)

    return pcchat


def get_deep_histograms(data, deep_data, key, cells, overlap_weighted_pzc, bins, overlap_key='overlap_weight',
                        deep_som_size=64 * 64, deep_map_shape=(64 * 64,), interpolate_kwargs={}):
    """Return individual deep histograms for each cell. Can interpolate for empty cells.

    Parameters
    ----------
    deep_data             : cosmos data used here for Y3
    key                   : Parameter to extract from dataframe
    cells                 : A list of deep cells to return sample from, or a single int.
    overlap_weighted_pzc  : Use overlap_weights in p(z|c) histogram if True. Also required if you want to bin conditionalize
    overlap_key           : column name for the overlap weights in the dataframe, default to 'overlap_weight'
    bins                  : Bins we histogram the values into
    interpolate_kwargs    : arguments to pass in for performing interpolation between cells for redshift hists using a 2d gaussian of sigma scale_length out to max_length cells away.
    The two kwargs are    : 'scale_length' and 'max_length'
    Returns
    -------
    hists : a histogram of the values from self.data[key] for each deep cell
    """

    if len(interpolate_kwargs) > 0:  # pragma: no cover
        cells_keep = cells
        cells = np.arange(deep_som_size)
    else:
        cells_keep = cells

    hists = []
    missing_cells = []
    populated_cells = []

    for ci, c in enumerate(cells):
        try:
            df = deep_data.groupby('cell_deep').get_group(c)
            if type(key) is str:
                z = df[key].values
                if overlap_weighted_pzc:  # pragma: no cover
                    weights = df[overlap_key].values
                else:
                    weights = np.ones(len(z))
                hist = np.histogram(z, bins, weights=weights, density=True)[
                    0]  # make weighted histogram by overlap weights
                populated_cells.append([ci, c])
            elif type(key) is list:  # pragma: no cover
                # use full p(z)
                assert (bins is not None)
                hist = histogram_from_fullpz(df, key, overlap_weighted=overlap_weighted_pzc, bin_edges=bins)
            hists.append(hist)
        except KeyError as e:
            missing_cells.append([ci, c])
            hists.append(np.zeros(len(bins) - 1))
    hists = np.array(hists)

    if len(interpolate_kwargs) > 0:  # pragma: no cover
        # print('Interpolating {0} missing histograms'.format(len(missing_cells)))
        missing_cells = np.array(missing_cells)
        populated_cells = np.array(populated_cells)
        hist_conds = np.isin(cells, populated_cells[:, 1]) & np.all(np.isfinite(hists), axis=1)
        for ci, c in missing_cells:
            if c not in cells_keep:
                # don't worry about interpolating cells we won't use anyways
                continue

            central_index = np.zeros(len(deep_map_shape), dtype=int)
            # np.unravel_index(c, deep_map_shape, central_index)  # fills central_index
            cND = np.zeros(len(deep_map_shape), dtype=int)
            weight_map = np.zeros(deep_som_size)
            # gaussian_rbf(weight_map, central_index, cND, deep_map_shape, **interpolate_kwargs)  # fills weight_map
            hists[ci] = np.sum(hists[hist_conds] * (weight_map[hist_conds] / weight_map[hist_conds].sum())[:, None],
                               axis=0)

        # purge hists back to the ones we care about
        hists = hists[cells_keep]

    return hists


def histogram(data, deep_data, key, cells, cell_weights, pcchat, overlap_weighted_pzc, deep_som_size=64 * 64, bins=None,
              individual_chat=False, interpolate_kwargs={}):
    """Return histogram from values that live in specified wide cells by querying deep cells that contribute

    Parameters
    ----------
    key                  : Parameter(s) to extract from dataframe
    cells                : A list of wide cells to return sample from, or a single int.
    cell_weights         : How much we weight each wide cell. This is the array p(chat | sample)
    overlap_weighted_pzc : Weight contribution of galaxies within c by overlap_weight, if True. Weighting for p(c|chat) is done using stored transfer matrix.
    bins                 : Bins we histogram the values into
    individual_chat      : If True, compute p(z|chat) for each individual cell in cells. If False, compute a single p(z|{chat}) for all cells.
    interpolate_kwargs   : arguments to pass in for performing interpolation between cells for redshift hists using a 2d gaussian of sigma scale_length out to max_length cells away. The two kwargs are: 'scale_length' and 'max_length'

    Returns
    -------
    hist : a histogram of the values from self.data[key]

    Notes
    -----
    This method tries to marginalize wide assignments into what deep assignments it has

    """
    # get sample, p(z|c)
    all_cells = np.arange(deep_som_size)
    hists_deep = get_deep_histograms(data, deep_data, key=key, cells=all_cells,
                                     overlap_weighted_pzc=overlap_weighted_pzc,
                                     bins=bins, interpolate_kwargs=interpolate_kwargs)
    if individual_chat:  # then compute p(z|chat) for each individual cell in cells and return histograms
        hists = []
        for i, (cell, cell_weight) in enumerate(zip(cells, cell_weights)):
            # p(c|chat,s)p(chat|s) = p(c,chat|s)
            possible_weights = pcchat[:, [cell]] * np.array([cell_weight])[None]  # (n_deep_cells, 1)
            # sum_chat p(c,chat|s) = p(c|s)
            weights = np.sum(possible_weights, axis=-1)
            conds = (weights != 0) & np.all(np.isfinite(hists_deep), axis=1)
            # sum_c p(z|c) p(c|s) = p(z|s)
            hist = np.sum((hists_deep[conds] * weights[conds, None]), axis=0)

            dx = np.diff(bins)
            normalization = np.sum(dx * hist)
            if normalization != 0:
                hist = hist / normalization
            hists.append(hist)
        return hists
    else:  # compute p(z|{chat}) and return histogram
        # p(c|chat,s)p(chat|s) = p(c,chat|s)
        possible_weights = pcchat[:, cells] * cell_weights[None]  # (n_deep_cells, n_cells)
        # sum_chat p(c,chat|s) = p(c|s)
        weights = np.sum(possible_weights, axis=-1)
        conds = (weights != 0) & np.all(np.isfinite(hists_deep), axis=1)
        # sum_c p(z|c) p(c|s) = p(z|s)
        hist = np.sum((hists_deep[conds] * weights[conds, None]), axis=0)

        dx = np.diff(bins)
        normalization = np.sum(dx * hist)
        if normalization != 0:
            hist = hist / normalization
        return hist


def histogram_from_fullpz(df, key, overlap_weighted, bin_edges, full_pz_end=6.00, full_pz_npts=601):  # pragma: no cover
    """Preserve bins from Laigle"""
    dz_laigle = full_pz_end / (full_pz_npts - 1)
    condition = np.sum(~np.equal(bin_edges, np.arange(0 - dz_laigle / 2.,
                                                      full_pz_end + dz_laigle,
                                                      dz_laigle)))
    assert condition == 0

    single_cell_hists = np.zeros((len(df), len(key)))

    overlap_weights = np.ones(len(df))
    if overlap_weighted:
        overlap_weights = df['overlap_weight'].values

    single_cell_hists[:, :] = df[key].values

    # normalize sompz p(z) to have area 1
    dz = 0.01
    area = np.sum(single_cell_hists, axis=1) * dz
    area[area == 0] = 1  # some galaxies have pz with only one non-zero point. set these galaxies' histograms to have
    # area 1
    area = area.reshape(area.shape[0], 1)
    single_cell_hists = single_cell_hists / area

    # weight normalized p(z) by shear response
    single_cell_hists = np.multiply(overlap_weights, single_cell_hists.transpose()).transpose()

    # sum individual galaxy p(z) to single cell p(z)
    hist = np.sum(single_cell_hists, axis=0)

    # renormalize p(z|c)
    area = np.sum(hist) * dz
    hist = hist / area

    return hist


def redshift_distributions_wide(data,
                                deep_data,
                                overlap_weighted_pchat,
                                overlap_weighted_pzc,
                                bins,
                                pcchat,
                                deep_som_size=64 * 64,
                                tomo_bins={},
                                key='Z',
                                force_assignment=True,
                                interpolate_kwargs={}, **kwargs):
    """Returns redshift distribution for sample

    Parameters
    ----------
    data :  Data sample of interest with wide data
    deep_data: cosmos data
    overlap_weighted_pchat  : If True, use overlap weights for p(chat)
    overlap_weighted_pzc : If True, use overlap weights for p(z|c)
                Note that whether p(c|chat) is overlap weighted depends on how you built pcchat earlier.
    bins :      bin edges for redshift distributions data[key]
    tomo_bins : Which cells belong to which tomographic bins. First column is
                cell id, second column is an additional reweighting of galaxies in cell.
                If nothing is passed in, then we by default just use all cells
    key :       redshift key
    force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True
    interpolate_kwargs : arguments to pass in for performing interpolation
    between cells for redshift hists using a 2d gaussian of sigma
    scale_length out to max_length cells away. The two kwargs are:
    'scale_length' and 'max_length'

    Returns
    -------
    hists : Either a single array (if no tomo_bins) or multiple arrays

    """
    if len(tomo_bins) == 0:  # pragma: no cover
        cells, cell_weights = get_cell_weights_wide(data, overlap_weighted_pchat=overlap_weighted_pchat,
                                                    force_assignment=force_assignment, **kwargs)
        if cells.size == 0:
            hist = np.zeros(len(bins) - 1)
        else:
            hist = histogram(data, deep_data, key=key, cells=cells, cell_weights=cell_weights,
                             overlap_weighted_pzc=overlap_weighted_pzc, bins=bins,
                             interpolate_kwargs=interpolate_kwargs)
        return hist
    else:
        cells, cell_weights = get_cell_weights_wide(data, overlap_weighted_pchat,
                                                    force_assignment=force_assignment, **kwargs)
        cellsort = np.argsort(cells)
        cells = cells[cellsort]
        cell_weights = cell_weights[cellsort]

        # break up hists into the different bins
        hists = []
        for tomo_key in tomo_bins:
            cells_use = tomo_bins[tomo_key][:, 0]
            cells_binweights = tomo_bins[tomo_key][:, 1]
            cells_conds = np.searchsorted(cells, cells_use, side='left')
            if len(cells_conds) == 0:  # pragma: no cover
                hist = np.zeros(len(bins) - 1)
            else:
                hist = histogram(data, deep_data, key=key, cells=cells[cells_conds],
                                 cell_weights=cell_weights[cells_conds] * cells_binweights,
                                 pcchat=pcchat,
                                 deep_som_size=deep_som_size,
                                 overlap_weighted_pzc=overlap_weighted_pzc,
                                 bins=bins,
                                 interpolate_kwargs=interpolate_kwargs)
            hists.append(hist)
        hists = np.array(hists)
        return hists

def get_cell_weights(data, overlap_weighted, key):
    """Given data, get cell weights and indices

    Parameters
    ----------
    data :  Dataframe we extract parameters from
    overlap_weighted : If True, use mean overlap weights of cells.
    key :   Which key we are grabbing

    Returns
    -------
    cells :         The names of the cells
    cell_weights :  The fractions of the cells
    """
    if overlap_weighted:  # pragma: no cover
        cws = data.groupby(key)['overlap_weight'].sum()
    else:
        cws = data.groupby(key).size()

    cells = cws.index.values.astype(int)
    cws = cws / cws.sum()

    cell_weights = cws.values
    return cells, cell_weights


def get_cell_weights_wide(data, overlap_weighted_pchat, cell_key='cell_wide', force_assignment=False, **kwargs):
    """Given data, get cell weights p(chat) and indices from wide SOM

    Parameters
    ----------
    data             : Dataframe we extract parameters from
    overlap_weighted_pchat : If True, use mean overlap weights of wide cells in p(chat)
    cell_key         : Which key we are grabbing. Default: cell_wide
    force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True

    Returns
    -------
    cells        :  The names of the cells
    cell_weights :  The fractions of the cells
    """
    # if force_assignment:
    #     data[cell_key] = self.assign_wide(data, **kwargs)
    return get_cell_weights(data, overlap_weighted_pchat, cell_key)


def define_tomo_bins_modal_spec(spec_data, deep_som_size, wide_som_size, bin_edges,
                                key_z='Z', key_cells_wide='cell_wide_unsheared'):
    """
    Assign each galaxy in our spec sample to a tomographic bin according to 
    some arbitrary bin edges such that each bin have a similar number of galaxies, 
    and assign each wide som cell to the bin to which a plurality of its constituent 
    spec galaxies are assigned. In practice, bin edges may be tuned to yield
    tomographic bins with roughly equal numbers of galaxies.
    """
    # assign gals in redshift sample to bins using arbitrary bin edges
    xlabels = []
    nbins = len(bin_edges) - 1
    for ii in range(nbins):
        xlabels.append(ii)
    # identify tomographic bin (redshift slice) for each galaxy in spec_data
    spec_data['tomo_bin'] = pd.cut(spec_data[key_z], bin_edges, labels=xlabels)  

    # assign the wide SOM cell to the tomographic bin to which a plurality of cell spec_data are assigned
    # ncells_with_spec_data = len(np.unique(spec_data[key_cells_wide].values))
    cell_bin_assignment = np.ones(wide_som_size, dtype=int) * -1
    cells_with_spec_data = np.unique(spec_data[key_cells_wide].values)

    groupby_obj_value_counts = spec_data.groupby(key_cells_wide)['tomo_bin'].value_counts()

    for c in cells_with_spec_data:
        # identify the tomographic bin to which the plurality of spec_data are assigned
        bin_assignment = groupby_obj_value_counts.loc[c].index[0]
        cell_bin_assignment[c] = bin_assignment

    # reformat bins into dict
    tomo_bins_wide = {}
    nbins = len(bin_edges) - 1
    for i in range(nbins):
        tomo_bins_wide[i] = np.where(cell_bin_assignment == i)[0]

    return tomo_bins_wide

def define_tomo_bins_deep(data, deep_som_shape, overlap_weighted, n_bins=5, key='Z',
                          cell_key='cell_deep', from_val=None, # force_assignment=True,
                          fullpzbins = np.arange(-0.005, 6.01, 0.01), interpolate_kwargs={}):
    """Returns which bins go into which tomographic sample. We order sample by key and the add cells until we have 1 / n_bins of the sample.

    Parameters
    ----------
    data :      Data sample of interest with deep data
    overlap_weighted : Use overlap weights for tomo bin definition, i.e. in p(z|c)
    n_bins :    Number of tomographic bins
    key :       Key that we use to order cells
    cell_key :   Which key we are grabbing. Default: cell_deep
    force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True
    from_val :  Minimum value of binning
    interpolate_kwargs : arguments to pass in for performing interpolation between cells for mean spec redshift using a 2d gaussian of sigma scale_length out to max_length cells away. The two kwargs are: 'scale_length' and 'max_length'

    Returns
    -------
    deep_bins : A dictionary of deep cell assignments

    """

    deep_som_size = np.prod(deep_som_shape)
    cell_indices = np.arange(deep_som_size)  # this can probably be done in a smarter fashion
    cell_assignments = np.zeros(deep_som_size, dtype=int) - 1

    # get mean z of spec data
    _deep_groups = data.groupby(cell_key)
    spec_cells = _deep_groups.size().index.values
    if type(key) is str:
        spec_cells_z = _deep_groups.agg('mean')[key].values
        mean_z_c = np.zeros(deep_som_size) + np.nan
        mean_z_c[spec_cells] = spec_cells_z
    elif type(key) is list:
        spec_cells_pz = get_deep_histograms(key, spec_cells, overlap_weighted_pzc=overlap_weighted, bins=fullpzbins)
        #spec_cells_z = np.array([np.sum(fullpzbins * hist) / np.sum(hist) if (np.sum(hist) > 0) else np.nan for hist in spec_cells_pz])
        spec_cells_z = np.array([np.sum((hist / np.sum(hist)) * (fullpzbins[1:] + fullpzbins[:-1]) / 2.) if (np.sum(hist) > 0) else np.nan for hist in spec_cells_pz])
        mean_z_c = np.zeros(deep_som_size) + np.nan
        mean_z_c[spec_cells] = spec_cells_z

    if len(spec_cells) != deep_som_size:
        print('Warning! We have {0} deep cells, but our spec sample only occupies {1}! We are {3} {2} cells'.format(deep_som_size, len(spec_cells), deep_som_size - len(spec_cells), ['cutting out', 'interpolating'][len(interpolate_kwargs) > 0]))
        if len(interpolate_kwargs) > 0:
            # get which cells are missing spec
            missing_cells = cell_indices[np.isin(cell_indices, spec_cells, invert=True)]
            for c in missing_cells:
                central_index = np.zeros(len(deep_som_shape), dtype=int)
                np.unravel_index(c, deep_som_shape, central_index)
                cND = np.zeros(len(deep_som_shape), dtype=int)
                weight_map = np.zeros(deep_som_size)
                gaussian_rbf(weight_map, central_index, cND, deep_som_shape, **interpolate_kwargs)
                mean_z_c[c] = np.sum(mean_z_c[spec_cells] * weight_map[spec_cells] / weight_map[spec_cells].sum())

    # get occupation of cells from your data
    sample_cells, sample_cell_weights = get_cell_weights(data, overlap_weighted, cell_key)
    # OK to not be overlap_weighted - will only use for occupation statistics
    sample_occupation = np.zeros(deep_som_size)
    sample_occupation[sample_cells] = sample_cell_weights

    # rank sort by mean z
    ordering_all = np.argsort(mean_z_c)  # nan to go end of the list
    # cut from ordering the nans
    ordering = ordering_all[np.isfinite(mean_z_c[ordering_all])]
    if from_val != None:
        cells_in_bin_0 = ordering[mean_z_c[ordering] < from_val]
        cell_assignments[cells_in_bin_0] = 0
        ordering = ordering[mean_z_c[ordering] >= from_val]

    # cumsum the occupation
    cumsum_occupation = np.cumsum(sample_occupation[ordering])
    # OK to not be overlap_weighted - will only use for occupation statistics
    if cumsum_occupation[-1] < 1.:
        print('Warning! We only have {0} of the sample in {1} cells with spec_z.'.format(cumsum_occupation[-1], len(ordering)))
        cumsum_occupation = cumsum_occupation / cumsum_occupation[-1]
    ordered_indices = cell_indices[ordering]

    if from_val==None:
        j=0
    else:
        j=1
    # assign to groups based on percentile
    for i in np.arange(j, n_bins, 1):
        lower = (i-j) / (n_bins-j)
        upper = (i + 1-j) / (n_bins-j)
        conds = (cumsum_occupation >= lower) * (cumsum_occupation <= upper)
        if upper==1:
            conds = (cumsum_occupation >= lower)
        cells_in_bin = ordered_indices[conds]
        cell_assignments[cells_in_bin] = i

    # convert into tomo_bins
    tomo_bins = {}
    for i in np.unique(cell_assignments):
        tomo_bins[i] = np.where(cell_assignments == i)[0]
    return tomo_bins


def define_tomo_bins_deep_fast(data, deep_som_shape, overlap_weighted, n_bins=5, key='Z',
                               cell_key='cell_deep', from_val=None,
                               fullpzbins=np.arange(-0.005, 6.01, 0.01), interpolate_kwargs={},
                               pz_c=None, zbins=None):
    """Fast variant of define_tomo_bins_deep using vectorized bincounts.

    This function is a drop-in replacement for define_tomo_bins_deep and keeps
    the same output format. It can optionally use precomputed p(z|c) to avoid
    recomputing per-cell mean redshifts.
    """
    # Accept either an integer SOM size or a 2D SOM shape.
    if np.isscalar(deep_som_shape):
        deep_som_shape_tuple = None
        deep_som_size = int(deep_som_shape)
    else:
        deep_som_shape_tuple = tuple(np.asarray(deep_som_shape, dtype=int))
        deep_som_size = int(np.prod(deep_som_shape_tuple))

    cell_indices = np.arange(deep_som_size)
    cell_assignments = np.zeros(deep_som_size, dtype=int) - 1

    # Use already-assigned cells directly instead of groupby.
    cell_ids_all = np.asarray(data[cell_key], dtype=np.int64)
    valid_cells_mask = (cell_ids_all >= 0) & (cell_ids_all < deep_som_size)
    cell_ids = cell_ids_all[valid_cells_mask]

    # Build per-cell mean redshift.
    mean_z_c = np.zeros(deep_som_size, dtype=float) + np.nan
    if pz_c is not None and zbins is not None:
        pz_c_arr = np.asarray(pz_c, dtype=float)
        z_centers = 0.5 * (np.asarray(zbins)[1:] + np.asarray(zbins)[:-1])
        if pz_c_arr.shape[0] != deep_som_size:
            raise ValueError(f"pz_c has {pz_c_arr.shape[0]} cells, expected {deep_som_size}")
        if pz_c_arr.shape[1] != z_centers.shape[0]:
            raise ValueError(f"pz_c has {pz_c_arr.shape[1]} zbins, expected {z_centers.shape[0]}")
        denom = np.sum(pz_c_arr, axis=1)
        good = denom > 0
        mean_z_c[good] = np.sum(pz_c_arr[good] * z_centers[None, :], axis=1) / denom[good]
        spec_cells = np.where(good)[0]
    elif isinstance(key, str):
        zvals = np.asarray(data[key])[valid_cells_mask]
        zsum = np.bincount(cell_ids, weights=zvals, minlength=deep_som_size)
        zcount = np.bincount(cell_ids, minlength=deep_som_size).astype(float)
        good = zcount > 0
        mean_z_c[good] = zsum[good] / zcount[good]
        spec_cells = np.where(good)[0]
    else:
        # Keep compatibility with non-scalar key mode by delegating.
        return define_tomo_bins_deep(
            data,
            deep_som_shape,
            overlap_weighted=overlap_weighted,
            n_bins=n_bins,
            key=key,
            cell_key=cell_key,
            from_val=from_val,
            fullpzbins=fullpzbins,
            interpolate_kwargs=interpolate_kwargs,
        )

    # Optionally interpolate missing deep-cell means if a 2D shape is available.
    if len(spec_cells) != deep_som_size:
        print('Warning! We have {0} deep cells, but our spec sample only occupies {1}! We are {3} {2} cells'.format(deep_som_size, len(spec_cells), deep_som_size - len(spec_cells), ['cutting out', 'interpolating'][len(interpolate_kwargs) > 0]))
        if len(interpolate_kwargs) > 0 and deep_som_shape_tuple is not None:
            missing_cells = cell_indices[np.isin(cell_indices, spec_cells, invert=True)]
            for c in missing_cells:
                central_index = np.zeros(len(deep_som_shape_tuple), dtype=int)
                np.unravel_index(c, deep_som_shape_tuple, central_index)
                cND = np.zeros(len(deep_som_shape_tuple), dtype=int)
                weight_map = np.zeros(deep_som_size)
                gaussian_rbf(weight_map, central_index, cND, deep_som_shape_tuple, **interpolate_kwargs)
                denom = weight_map[spec_cells].sum()
                if denom > 0:
                    mean_z_c[c] = np.sum(mean_z_c[spec_cells] * weight_map[spec_cells] / denom)
        elif len(interpolate_kwargs) > 0 and deep_som_shape_tuple is None:
            print("Warning! interpolate_kwargs provided but deep_som_shape is scalar; skipping interpolation.")

    # Fast per-cell occupation using existing cell assignments.
    if overlap_weighted and 'overlap_weight' in data:
        occ_weights = np.asarray(data['overlap_weight'])[valid_cells_mask]
        occ_counts = np.bincount(cell_ids, weights=occ_weights, minlength=deep_som_size).astype(float)
    else:
        occ_counts = np.bincount(cell_ids, minlength=deep_som_size).astype(float)
    total_occ = occ_counts.sum()
    sample_occupation = occ_counts / total_occ if total_occ > 0 else occ_counts

    # Rank cells by mean z and assign tomographic bins by cumulative occupation.
    ordering_all = np.argsort(mean_z_c)
    ordering = ordering_all[np.isfinite(mean_z_c[ordering_all])]
    if from_val is not None:
        cells_in_bin_0 = ordering[mean_z_c[ordering] < from_val]
        cell_assignments[cells_in_bin_0] = 0
        ordering = ordering[mean_z_c[ordering] >= from_val]

    if ordering.size > 0:
        cumsum_occupation = np.cumsum(sample_occupation[ordering])
        if cumsum_occupation[-1] < 1.:
            print('Warning! We only have {0} of the sample in {1} cells with spec_z.'.format(cumsum_occupation[-1], len(ordering)))
            if cumsum_occupation[-1] > 0:
                cumsum_occupation = cumsum_occupation / cumsum_occupation[-1]
        ordered_indices = cell_indices[ordering]

        j = 0 if from_val is None else 1
        for i in np.arange(j, n_bins, 1):
            lower = (i - j) / (n_bins - j)
            upper = (i + 1 - j) / (n_bins - j)
            conds = (cumsum_occupation >= lower) * (cumsum_occupation <= upper)
            if upper == 1:
                conds = (cumsum_occupation >= lower)
            cells_in_bin = ordered_indices[conds]
            cell_assignments[cells_in_bin] = i

    tomo_bins = {}
    for i in np.unique(cell_assignments):
        tomo_bins[i] = np.where(cell_assignments == i)[0]
    return tomo_bins


def define_tomo_bins_wide(pc_chat, deep_bins, dfilter=0.0):
    """Returns which wide bins go into which tomographic sample.

    Parameters
    ----------
    deep_bins : A dictionary of deep cell assignments
    dfilter :   Require the probability of belonging to this bin to be
                dfilter more likely than the second most likely bin.
                Default is to disable this with dfilter = 0.0

    Returns
    -------
    wide_bins : Same as deep, only for wide assignments

    Notes
    -----
    For each deep bin, calculates sum_{c \in bin b} p(c | chat)
    chat then goes into bin b with largest value
    """

    keys = deep_bins.keys()
    # sum_{c\in bin} p(c|chat)
    probabilities = []
    for key in keys:
        cells = deep_bins[key]
        prob = np.sum(pc_chat[cells], axis=0)
        probabilities.append(prob)
    probabilities = np.array(probabilities)

    # dfilter
    sorted_probabilities = np.sort(probabilities, axis=0)
    dprob = sorted_probabilities[-1] - sorted_probabilities[-2]

    # group assignments
    # binhat = argmax_bin sum_{c\in bin} p(c|chat)
    assignments = np.argmax(probabilities, axis=0)
    wide_bins = {}
    for key_i, key in enumerate(keys):
        wide_bins[key] = np.where((assignments == key_i) * (dprob >= dfilter))[0]
    return wide_bins

def nz_bin_conditioned(wide_data, spec_data, overlap_weighted_pchat, overlap_weighted_pzc, tomo_cells, zbins, pcchat,
                       cell_wide_key='cell_wide', zkey='Z'):
    f"""Function to obtain p(z|bin,s): the redshift distribution of a tomographic bin
        including the tomographic selection effect in p(z|chat). 
        The truth: n(z| s-hat) = sum_{{c,c-hat}} p(z|c, chat, s-hat, ) p(c | chat, s-hat) p(c-hat)
        is approximated  here with : n(z| s-hat) = sum_{{c,c-hat}} p(z|c, bhat, s-hat, ) p(c | chat, s-hat) p(c-hat)
        which is closer to the truth than vanilla SOMPZ: n(z| s-hat) = sum_{{c,c-hat}} p(z|c, s-hat, ) p(c | chat, s-hat) p(c-hat)

    Implementation note:
    This is going to sneak the bin conditionalization into the overlap weights, and then divide them back out.
    This is a simple way of achieving to not completely lose cells c that contribute to p(c|chat) but don't have a z in b.

        Parameters
        ----------
        wide_data : Wide field data, pandas DataFrame
        spec_data : Spectroscopic calibration data, pandas DataFrame
        overlap_weighted_pchat : If True, weight chat by the sum of overlap weights, not number of galaxies, in wide field data.
        tomo_cells : Which cells belong to this tomographic bin. First column is
                     cell id, second column is an additional reweighting of galaxies in that cell.
        zbins : redshift bin edges.
        cell_wide_key : key for wide SOM cell id information in spec_data.
        cell_deep_key : key for wide SOM cell id information in spec_data.
    """

    print('Ngal full redshift sample:', len(spec_data))
    bl = len(spec_data[spec_data['cell_wide_unsheared'].isin(tomo_cells[:, 0])])
    print('Ngal subset of redshift sample in bin:', bl)

    f = 1.e9  # how much more we weight the redshift of a galaxy that's in the right bin

    stored_overlap_weight = spec_data['overlap_weight'].copy()  # save for later

    if not overlap_weighted_pzc: # we must set values for overlap_weight, even if the user doesn't want overlap weighting
        spec_data['overlap_weight'] = np.ones(len(spec_data))

    spec_data.loc[spec_data['cell_wide_unsheared'].isin(tomo_cells[:, 0]), 'overlap_weight'] *= f

    nz = redshift_distributions_wide(data=wide_data, deep_data=spec_data, 
                                     overlap_weighted_pchat=overlap_weighted_pchat,
                                     overlap_weighted_pzc=True, # hard-coded here
                                     bins=zbins, pcchat=pcchat, 
                                     tomo_bins={"mybin": tomo_cells}, key=zkey,
                                     force_assignment=False, 
                                     cell_key=cell_wide_key)

    spec_data['overlap_weight'] = stored_overlap_weight.copy()  # open jar

    return nz[0]

def tomo_bins_wide_2d(tomo_bins_wide_dict):
    tomo_bins_wide = tomo_bins_wide_dict.copy()
    for k in tomo_bins_wide:
        if tomo_bins_wide[k].ndim == 1:
            tomo_bins_wide[k] = np.column_stack((tomo_bins_wide[k], np.ones(len(tomo_bins_wide[k]))))
        renorm = 1. / np.average(tomo_bins_wide[k][:, 1])
        tomo_bins_wide[k][:, 1] *= renorm  # renormalize so the mean weight is 1; important for bin conditioning
    return tomo_bins_wide

class SOMPZInformer(CatInformer):
    """Inform stage for SOMPZEstimator
    """
    name = "SOMPZInformer"
    config_options = CatInformer.config_options.copy()
    config_options.update(redshift_col=SHARED_PARAMS,
                          hdf5_groupname=SHARED_PARAMS,
                          nprocess=Param(int, 1, msg="number of processors to use"),
                          # groupname=Param(str, "photometry", msg="hdf5_groupname for ata"),
                          inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for data"),
                          input_errs=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for data"),
                          zero_points=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for data, if needed"),
                          som_shape=Param(list, [32, 32], msg="shape for the som, must be a 2-element tuple"),
                          som_minerror=Param(float, 0.01, msg="floor placed on observational error on each feature in som"),
                          som_wrap=Param(bool, False, msg="flag to set whether the SOM has periodic boundary conditions"),
                          som_take_log=Param(bool, True, msg="flag to set whether to take log of inputs (i.e. for fluxes) for som"),
                          convert_to_flux=Param(bool, False, msg="flag for whether to convert input columns to fluxes for data"
                                                "set to true if inputs are mags and to False if inputs are already fluxes"),
                          set_threshold=Param(bool, False, msg="flag for whether to replace values below a threshold with a set number"),
                          thresh_val=Param(float, 1.e-5, msg="threshold value for set_threshold for data"),
                          thresh_val_err=Param(float, 1.e-5, msg="threshold value for set_threshold for data error"))

    inputs = [('input_data', TableHandle),
              ]
    outputs = [('model', ModelHandle),
               ]

    def run(self):
        # print(f'Using SOMPZInformer from file {inspect.getfile(SOMPZInformer)}')
        if self.config.hdf5_groupname:  # pragma: no cover
            data = self.get_data('input_data')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            # DEAL with hdf5_groupname stuff later, just assume it's in the top level for now!
            data = self.get_data('input_data')
        num_inputs = len(self.config.inputs)
        
        ngal = len(data[self.config.inputs[0]])
        print(f"{ngal} galaxies in sample")

        d_input = np.zeros([ngal, num_inputs])
        d_errs = np.zeros([ngal, num_inputs])

        # assemble data
        for i, (col, errcol) in enumerate(zip(self.config.inputs, self.config.input_errs)):
            if self.config.convert_to_flux:
                d_input[:, i] = mag2flux(data[col], self.config.zero_points[i])
                d_errs[:, i] = magerr2fluxerr(data[errcol], d_input[:, i])
            else:  # pragma: no cover
                d_input[:, i] = data[col]
                d_errs[:, i] = data[errcol]

        # for each feature, set values below a threshold to a value set by the config
        if self.config.set_threshold:
            for i in range(num_inputs):
                mask = (d_input[:, i] < self.config.thresh_val)
                d_input[:, i][mask] = self.config.thresh_val
                errmask = (d_errs[:, i] < self.config.thresh_val_err)
                d_errs[:, i][errmask] = self.config.thresh_val_err

        sommetric = somfuncs.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
        learn_func = somfuncs.hFunc(ngal, sigma=(30, 1))

        # if 'pool' in self.config.keys():
        #     self.pool, self.nprocess = self.config["pool"]
        # else:
        #     print("pool not specified, setting pool to None")
        #     self.pool = None
        #     self.nprocess = 0
        #     self.config.pool = (None, 1)
        pool = Pool(self.config.nprocess)
        nprocess = self.config.nprocess
        pooltuple = (pool, nprocess)

        print(f"Training SOM of shape {self.config.som_shape} with pool made of {nprocess} processes...", flush=True)

        som = somfuncs.NoiseSOM(sommetric, d_input, d_errs, learn_func,
                                shape=self.config.som_shape, minError=self.config.som_minerror,
                                wrap=self.config.som_wrap, logF=self.config.som_take_log, pool=pooltuple)
        model = dict(som=som, columns=self.config.inputs,
                     err_columns=self.config.input_errs)
        self.add_data('model', model)

    def inform(self, input_data):
        self.set_data('input_data', input_data)
        self.run()
        self.finalize()
        return self.model


class SOMPZEstimator(CatEstimator):  # pragma: no cover
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "SOMPZEstimator"
    config_options = CatEstimator.config_options.copy()
    config_options.update(redshift_col=SHARED_PARAMS,
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
                          # data_path=Param(str, "directory", msg="directory for output files"),
                          spec_groupname=Param(str, "photometry", msg="hdf5_groupname for spec_data"),
                          balrog_groupname=Param(str, "photometry", msg="hdf5_groupname for balrog_data"),
                          wide_groupname=Param(str, "photometry", msg="hdf5_groupname for wide_data"),
                          specz_name=Param(str, "redshift", msg="column name for true redshift in specz sample"),
                          inputs_deep=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          input_errs_deep=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for deep data"),
                          inputs_wide=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for wide data"),
                          input_errs_wide=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for wide data"),
                          zero_points_deep=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for deep data, if needed"),
                          zero_points_wide=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for wide data, if needed"),
                          som_shape_deep=Param(list, [32, 32], msg="shape for the deep som, must be a 2-element tuple"),
                          som_shape_wide=Param(list, [32, 32], msg="shape for the wide som, must be a 2-element tuple"),
                          som_minerror_deep=Param(float, 0.01, msg="floor placed on observational error on each feature in deep som"),
                          som_minerror_wide=Param(float, 0.01, msg="floor placed on observational error on each feature in wide som"),
                          som_wrap_deep=Param(bool, False, msg="flag to set whether the deep SOM has periodic boundary conditions"),
                          som_wrap_wide=Param(bool, False, msg="flag to set whether the wide SOM has periodic boundary conditions"),
                          som_take_log_deep=Param(bool, True, msg="flag to set whether to take log of inputs (i.e. for fluxes) for deep som"),
                          som_take_log_wide=Param(bool, True, msg="flag to set whether to take log of inputs (i.e. for fluxes) for wide som"),
                          convert_to_flux_deep=Param(bool, False, msg="flag for whether to convert input columns to fluxes for deep data"
                                                     "set to true if inputs are mags and to False if inputs are already fluxes"),
                          convert_to_flux_wide=Param(bool, False, msg="flag for whether to convert input columns to fluxes for wide data"),
                          set_threshold_deep=Param(bool, False, msg="flag for whether to replace values below a threshold with a set number"),
                          thresh_val_deep=Param(float, 1.e-5, msg="threshold value for set_threshold for deep data"),
                          set_threshold_wide=Param(bool, False, msg="flag for whether to replace values below a threshold with a set number"),
                          thresh_val_wide=Param(float, 1.e-5, msg="threshold value for set_threshold for wide data"),
                          overlap_weighted_pchat=Param(bool, False, msg="if True, use overlap_weight for p(chat)"),
                          overlap_weighted_pzc=Param(bool, False, msg="if True, use overlap_weight for p(z|c)"),
                          overlap_weighted=Param(bool, False, msg="if True, use overlap_weight when defining tomographic bins"),
                          debug=Param(bool, False, msg="boolean reducing dataset size for quick debuggin"))

    inputs = [('deep_model', ModelHandle),
              ('wide_model', ModelHandle),
              ('spec_data', TableHandle),
              ('balrog_data', TableHandle),
              ('wide_data', TableHandle)]
    outputs = [('nz', QPHandle),
               ('spec_data_deep_assignment', Hdf5Handle),
               ('spec_data_wide_assignment', Hdf5Handle),
               ('balrog_data_deep_assignment', Hdf5Handle),
               ('balrog_data_wide_assignment', Hdf5Handle),
               ('wide_data_assignment', Hdf5Handle),
               ('pz_c', Hdf5Handle),
               ('pz_chat', Hdf5Handle),
               ('pc_chat', Hdf5Handle),
               ]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        print(f'Using SOMPZEstimator from file {inspect.getfile(SOMPZEstimator)}')
        if 'pool' in self.config.keys():
            self.pool, self.nprocess = self.config["pool"]
        else:
            self.pool = None
            self.nprocess = 0
        # check on bands, errs, and prior band
        if len(self.config.inputs_deep) != len(self.config.input_errs_deep):  # pragma: no cover
            raise ValueError("Number of inputs_deep specified in inputs_deep must be equal to number of mag errors specified in input_errs_deep!")
        if len(self.config.inputs_wide) != len(self.config.input_errs_wide):  # pragma: no cover
            raise ValueError("Number of inputs_wide specified in inputs_wide must be equal to number of mag errors specified in input_errs_wide!")

    def open_model(self, **kwargs):
        """Load the model and/or attach it to this Creator.

        Keywords
        --------
        model : object, str or ModelHandle
            Either an object with a trained model, a path pointing to a file
            that can be read to obtain the trained model, or a ``ModelHandle``
            providing access to the trained model

        Returns
        -------
        self.model : object
            The object encapsulating the trained model
        """
        deep_model = kwargs.get("deep_model", None)
        wide_model = kwargs.get("wide_model", None)
        if deep_model is None or deep_model == "None":  # pragma: no cover
            self.deep_model = None
        else:
            if isinstance(deep_model, str):  # pragma: no cover
                self.deep_model = self.set_data("deep_model", data=None, path=deep_model)
                self.config["deep_model"] = deep_model
            else:
                if isinstance(deep_model, ModelHandle):  # pragma: no cover
                    if deep_model.has_path:
                        self.config["deep_model"] = deep_model.path
                self.deep_model = self.set_data("deep_model", deep_model)

        if wide_model is None or wide_model == "None":  # pragma: no cover
            self.wide_model = None
        else:
            if isinstance(wide_model, str):  # pragma: no cover
                self.wide_model = self.set_data("wide_model", data=None, path=wide_model)
                self.config["wide_model"] = wide_model
            else:
                if isinstance(wide_model, ModelHandle):  # pragma: no cover
                    if wide_model.has_path:
                        self.config["wide_model"] = wide_model.path
                self.wide_model = self.set_data("wide_model", wide_model)
        return self.deep_model, self.wide_model

    def _assign_som(self, flux, flux_err, somstr):
        if somstr == 'deep':
            som_dim = self.config.som_shape_deep[0]
        elif somstr == 'wide':
            som_dim = self.config.som_shape_wide[0]

        nTrain = flux.shape[0]
        if somstr == "deep":
            som_weights = self.deep_model['som'].weights
        elif somstr == "wide":
            som_weights = self.wide_model['som'].weights
        else:
            # assert (0)
            raise ValueError(f"valid SOM values are 'deep' and 'wide', {somstr} is not valid")
        hh = somfuncs.hFunc(nTrain, sigma=(30, 1))
        metric = somfuncs.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
        som = somfuncs.NoiseSOM(metric, None, None,
                                learning=hh,
                                shape=(som_dim, som_dim),
                                wrap=False, logF=True,
                                initialize=som_weights,
                                minError=0.02)
        subsamp = 1

        # Now we classify the objects into cells and save these cells
        if self.pool is not None:
            inds = np.array_split(np.arange(len(flux)), self.nprocess)
            pickableclassify = Pickableclassify(som, flux, flux_err, inds)
            result = self.pool.map(pickableclassify, range(self.nprocess))
            cells_test = np.concatenate([r[0] for r in result])
            dist_test = np.concatenate([r[1] for r in result])
            del pickableclassify
        else:
            cells_test, dist_test = som.classify(flux[::subsamp, :], flux_err[::subsamp, :])

        # take out numpy savez
        # outfile = os.path.join(output_path, "som_{0}_{1}x{1}_assign.npz".format(somstr,som_dim))
        # np.savez(outfile, cells=cells_test, dist=dist_test)

        return cells_test, dist_test

    def _estimate_pdf(self,):
        zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2.,
                          self.config.zbins_max + self.config.zbins_dz,
                          self.config.zbins_dz)
        self.bincents = 0.5 * (zbins[1:] + zbins[:-1])
        
        deep_som_size = np.product(self.deep_model['som'].shape)
        wide_som_size = np.product(self.wide_model['som'].shape)

        all_deep_cells = np.arange(deep_som_size)
        # key = 'specz_redshift'
        key = self.config.specz_name

        # load spec_data to access redshifts to histogram

        # TODO: this code block is repeated in run. refactor to avoid repeating
        if self.config.spec_groupname:
            spec_data = self.get_data('spec_data')[self.config.spec_groupname]
        else:  # pragma: no cover
            # DEAL with hdf5_groupname stuff later, just assume it's in the top level for now!
            spec_data = self.get_data('spec_data')

        if self.config.balrog_groupname:
            balrog_data = self.get_data('balrog_data')[self.config.balrog_groupname]
        else:  # pragma: no cover
            balrog_data = self.get_data('balrog_data')
        if self.config.wide_groupname:
            wide_data = self.get_data('wide_data')[self.config.wide_groupname]
        else:  # pragma: no cover
            wide_data = self.get_data('wide_data')

        if self.config.debug:
            spec_data = spec_data[:2000]
            balrog_data = balrog_data[:2000]

        # spec_data = self.get_data('spec_data')
        # balrog_data = self.get_data('balrog_data')
        # wide_data = self.get_data('wide_data')

        cell_deep_spec_data = self.deep_assignment['spec_data'][0]
        cell_wide_spec_data = self.wide_assignment['spec_data'][0]
        cell_deep_balrog_data = self.deep_assignment['balrog_data'][0]
        cell_wide_balrog_data = self.wide_assignment['balrog_data'][0]

        spec_data_for_pz = pd.DataFrame({key: spec_data[key],
                                         'cell_deep': cell_deep_spec_data,
                                         'cell_wide': cell_wide_spec_data})
        if 'overlap_weight' in spec_data.keys(): # : dtype.names:
            spec_data_for_pz['overlap_weight'] = spec_data['overlap_weight']

        balrog_data_for_pz = pd.DataFrame({key: balrog_data[key],
                                         'cell_deep': cell_deep_balrog_data,
                                         'cell_wide': cell_wide_balrog_data})
        if 'overlap_weight' in balrog_data.keys():
            balrog_data_for_pz['overlap_weight'] = balrog_data['overlap_weight']

        # compute p(z|c, etc.), redshift histograms of deep SOM cells
        pz_c = np.array(get_deep_histograms(None,  # this arg is not currently used in get_deep_histograms
                                            spec_data_for_pz,
                                            key=key,
                                            cells=all_deep_cells,
                                            overlap_weighted_pzc=self.config.overlap_weighted_pzc,
                                            bins=zbins))
        # compute p(c|chat, etc.), the deep-wide transfer function
        pc_chat = calculate_pcchat(deep_som_size,
                                   wide_som_size,
                                   self.deep_assignment['balrog_data'][0],  # balrog_data['cell_deep'],#.values,
                                   self.wide_assignment['balrog_data'][0],  # balrog_data['cell_wide'],#.values,
                                   np.ones(len(self.deep_assignment['balrog_data'][0])))
        pcchatdict = dict(pc_chat=pc_chat)
        self.add_data('pc_chat', pcchatdict)
        
        # compute p(chat), occupation in wide SOM cells
        all_wide_cells = np.arange(wide_som_size)
        cell_wide_wide_data = self.wide_assignment['wide_data'][0]
        wide_data_for_pz = pd.DataFrame({'cell_wide': cell_wide_wide_data})
        if 'overlap_weight' in wide_data.keys():
            wide_data_for_pz['overlap_weight'] = wide_data['overlap_weight']

        # compute p(z|chat) \propto sum_c p(z|c) p(c|chat)
        pz_chat = np.array(histogram(wide_data_for_pz,
                                     spec_data_for_pz,
                                     key=key,
                                     pcchat=pc_chat,
                                     cells=all_wide_cells,
                                     cell_weights=np.ones(len(all_wide_cells)),
                                     deep_som_size=deep_som_size,
                                     overlap_weighted_pzc=self.config.overlap_weighted_pzc,
                                     bins=zbins,
                                     individual_chat=True))
        pzchatdict = dict(pz_chat=pz_chat)
        self.add_data('pz_chat', pzchatdict)

        # assign sample to tomographic bins
        # DES Y3-like tomographic binning
        # tomo_bins_wide_dict = define_tomo_bins_modal_spec(spec_data_for_pz,
        #                                           deep_som_size,
        #                                           wide_som_size,
        #                                           bin_edges=self.config.bin_edges,
        #                                           key_z=key,
        #                                           key_cells_wide='cell_wide')
        # BuchsDavis19-like tomographic binning
        tomo_bins_deep_dict = define_tomo_bins_deep(balrog_data_for_pz,
                                                    self.deep_model['som'].shape,
                                                    overlap_weighted=self.config.overlap_weighted,
                                                    n_bins=len(self.config.bin_edges)-1,
                                                    key=key,
                                                    cell_key='cell_deep')
        # tomo_bins_deep = tomo_bins_wide_2d(tomo_bins_deep_dict)
        tomo_bins_wide_dict = define_tomo_bins_wide(pc_chat, tomo_bins_deep_dict)
        tomo_bins_wide = tomo_bins_wide_2d(tomo_bins_wide_dict)
        # tomo_bins_deep = tomo_bins_wide_2d(tomo_bins_deep_dict)
        
        # compute number of galaxies per tomographic bin (diagnostic info)
        # cell_occupation_info = wide_data_for_pz.groupby('cell_wide')['cell_wide'].count()
        # bin_occupation_info = {'bin' + str(i) : np.sum(cell_occupation_info.loc[tomo_bins_wide_dict[i]].values) for i in range(n_bins)}
        # print(bin_occupation_info)

        # calculate n(z)
        nz = redshift_distributions_wide(data=wide_data_for_pz,
                                         deep_data=spec_data_for_pz,
                                         overlap_weighted_pchat=self.config.overlap_weighted_pchat,
                                         overlap_weighted_pzc=self.config.overlap_weighted_pzc,
                                         bins=zbins,
                                         deep_som_size=deep_som_size,
                                         pcchat=pc_chat,
                                         tomo_bins=tomo_bins_wide,
                                         key=key,
                                         force_assignment=False,
                                         cell_key='cell_wide')

        return tomo_bins_wide, pz_c, pc_chat, nz

    def _find_wide_tomo_bins(self, tomo_bins_wide):
        """the current code has a map of the wide galaxies to the wide SOM cells
        (in wide_assign), and the mapping of which wide som cells map to which tomo
        bin (in tomo_bins_wide, passed as an arg to this function), but not a direct
        map of which tomo bin each wide galaxy gets mapped to.  This function will
        return a dictionary with two entries, one that contains an array of the
        same length as the number of input galaxies that contains an integer
        corresponding to which tomographic bin the galaxy belongs to, and the other
        (weight) that corresponds to the weight associated with that bin in the
        tomo_bins_wide file (it looks to always be one, but I'll copy it just in
        case there are situations where it is not 1.0)

        Inputs: tomo_bins_wide (returned by estimate pdf)
        Returns: wide_tomo_bins (dict)
        """
        # This code does not handle the fact that som of the SOM has no balrog sample covered, thereby should not be included in the  calculation
        # assert (0)
        raise ValueError("this code is no longer used")
        wide_assign = self.widedict['cells']
        # print(tomo_bins_wide)

        # nbins = len(self.config.bin_edges)-1
        # ngal = len(wide_assign)
        # tomo_mask = np.zeros(ngal, dtype=int)
        tmp_cells = np.concatenate([tomo_bins_wide[nbin][:, 0].astype(np.int32) for nbin in tomo_bins_wide])
        tmp_weights = np.concatenate([tomo_bins_wide[nbin][:, 1] for nbin in tomo_bins_wide])
        tmp_bins = np.concatenate([(np.ones(len(tomo_bins_wide[nbin][:, 0])) * nbin).astype(int) for nbin in tomo_bins_wide])
        sortidx = np.argsort(tmp_cells)
        indices = sortidx[np.searchsorted(tmp_cells, wide_assign, sorter=sortidx)]
        tomo_bins = tmp_bins[indices]
        tomo_weights = tmp_weights[indices]

        tmask_dict = dict(bin=tomo_bins, weight=tomo_weights)
        return tmask_dict

    def _initialize_run(self):
        """
        code that gets run once
        """

        self._output_handle = None

    def _do_chunk_output(self):
        """
        code that gets run once
        """
        print('TODO')
        # assert False
        raise NotImplementedError("_do_chunk_output not yet implemented")

    def _finalize_run(self):
        self._output_handle.finalize_write()

    def _process_chunk(self, start, end, data, first):
        """
        Run SOMPZ on a chunk of data
        """
        ngal_wide = len(data[self.config.inputs_wide[0]])
        num_inputs_wide = len(self.config.inputs_wide)
        data_wide = np.zeros([ngal_wide, num_inputs_wide])
        data_err_wide = np.zeros([ngal_wide, num_inputs_wide])
        for j, (col, errcol) in enumerate(zip(self.config.inputs_wide, self.config.input_errs_wide)):
            if self.config.convert_to_flux_wide:
                data_wide[:, j] = mag2flux(np.array(data[col], dtype=np.float32), self.config.zero_points_wide[j])
                data_err_wide[:, j] = magerr2fluxerr(np.array(data[errcol], dtype=np.float32), data_wide[:, j])
            else:
                data_wide[:, j] = np.array(data[col], dtype=np.float32)
                data_err_wide[:, j] = np.array(data[errcol], dtype=np.float32)

        if self.config.set_threshold_wide:
            truncation_value = self.config.thresh_value_wide
            for j in range(num_inputs_wide):
                mask = (data_wide[:, j] < self.config.thresh_val_wide)
                data_wide[:, j][mask] = truncation_value
                errmask = (data_err_wide[:, j] < self.config.thresh_val_wide)
                data_err_wide[:, j][errmask] = truncation_value

        data_wide_ndarray = np.array(data_wide, copy=False)
        flux_wide = data_wide_ndarray.view()
        data_err_wide_ndarray = np.array(data_err_wide, copy=False)
        flux_err_wide = data_err_wide_ndarray.view()

        cells_wide, dist_wide = self._assign_som(flux_wide, flux_err_wide, 'wide')
        print('TODO store this info')
        output_handle = None
        self._do_chunk_output(output_handle, start, end, first)

    def run(self,):
        self.deep_model, self.wide_model = self.open_model(**self.config)  # None
        print('initialized model', self.deep_model, self.wide_model)
        if self.config.spec_groupname:
            spec_data = self.get_data('spec_data')[self.config.spec_groupname]
        else:  # pragma: no cover
            spec_data = self.get_data('spec_data')

        if self.config.balrog_groupname:
            balrog_data = self.get_data('balrog_data')[self.config.balrog_groupname]
        else:  # pragma: no cover
            balrog_data = self.get_data('balrog_data')

        if self.config.wide_groupname:
            wide_data = self.get_data('wide_data')[self.config.wide_groupname]
        else:  # pragma: no cover
            wide_data = self.get_data('wide_data')

        # iterator = self.input_iterator("wide_data")
        # first = True
        # self._initialize_run() # TODO implement
        # self._output_handle = None # TODO consider handle for dict to store all outputs
        # for s, e, data_chunk in iterator:
        #    if self.rank == 0:
        #        print(f"Process {self.rank} running estimator on chunk {s} - {e}")
        #    self._process_chunk(s, e, data_chunk, first)
        #    first = False
        #    gc.collect()

        # print('You need to do spec_data and balrog_data')
        # self._finalize_run()
        # assert False,'below this line is code that needs to be updated'

        samples = [spec_data, balrog_data, wide_data]
        # NOTE: DO NOT CHANGE NAMES OF 'labels' below! They are used
        # in the naming of the outputs of the stage!
        labels = ['spec_data', 'balrog_data', 'wide_data']
        # output_path = './' # make kwarg
        # assign samples to SOMs
        # TODO: handle case of sample already having been assigned
        self.deep_assignment = {}
        self.wide_assignment = {}
        for i, (data, label) in enumerate(zip(samples, labels)):
            print("Working on {0}\n".format(label), flush=True)
            if i <= 1:
                outlabel = f"{label}_deep_assignment"
                if os.path.isfile(self.config[outlabel]):
                    temp = h5py.File(self.config[outlabel], 'r')
                    cells_deep, dist_deep = temp['cells'][:], temp['dist'][:]
                    self.deep_assignment[label] = (cells_deep, dist_deep)
                    tmpdict = dict(cells=cells_deep, dist=dist_deep)
                    self.add_data(outlabel, tmpdict)
                    temp.close()
                else:
                    # print(self.config.inputs_deep)
                    # #######
                    #  REDO how subset of data is copied so that it works for hdf5
                    # data_deep = data[self.config.inputs_deep]
                    # data_deep_ndarray = np.array(data_deep,copy=False)
                    # Flux_deep = data_deep_ndarray.view((np.float32,
                    #                                    len(self.config.inputs_deep)))
                    ngal_deep = len(data[self.config.inputs_deep[0]])
                    num_inputs_deep = len(self.config.inputs_deep)
                    data_deep = np.zeros([ngal_deep, num_inputs_deep])
                    data_err_deep = np.zeros([ngal_deep, num_inputs_deep])
                    for j, (col, errcol) in enumerate(zip(self.config.inputs_deep, self.config.input_errs_deep)):
                        if self.config.convert_to_flux_deep:
                            data_deep[:, j] = mag2flux(np.array(data[col], dtype=np.float32), self.config.zero_points_deep[j])
                            data_err_deep[:, j] = magerr2fluxerr(np.array(data[errcol], dtype=np.float32), data_deep[:, j])
                        else:
                            data_deep[:, j] = np.array(data[col], dtype=np.float32)
                            data_err_deep[:, j] = np.array(data[errcol], dtype=np.float32)

                    # ### TRY PUTTING IN THRESHOLD FROM INFORM!
                    if self.config.set_threshold_deep:
                        truncation_value = self.config.thresh_val_deep
                        for j in range(num_inputs_deep):
                            mask = (data_deep[:, j] < self.config.thresh_val_deep)
                            data_deep[:, j][mask] = truncation_value
                            errmask = (data_err_deep[:, j] < self.config.thresh_val_deep)
                            data_err_deep[:, j][errmask] = truncation_value

                    data_deep_ndarray = np.array(data_deep, copy=False)
                    flux_deep = data_deep_ndarray.view()

                    # data_deep = data[self.config.err_inputs_deep]
                    # data_deep_ndarray = np.array(data_deep,copy=False)
                    # flux_err_deep = data_deep_ndarray.view((np.float32,
                    #                                         len(self.config.err_inputs_deep)))
                    data_err_deep_ndarray = np.array(data_err_deep, copy=False)
                    flux_err_deep = data_err_deep_ndarray.view()
                    cells_deep, dist_deep = self._assign_som(flux_deep, flux_err_deep, 'deep')

                    self.deep_assignment[label] = (cells_deep, dist_deep)
                    # take out numpy savez
                    # outfile = os.path.join(output_path, label + '_deep.npz')
                    # np.savez(outfile, cells=cells_deep, dist=dist_deep)
                    tmpdict = dict(cells=cells_deep, dist=dist_deep)
                    self.add_data(outlabel, tmpdict)
            else:
                cells_deep, dist_deep = None, None

            ngal_wide = len(data[self.config.inputs_wide[0]])
            num_inputs_wide = len(self.config.inputs_wide)
            data_wide = np.zeros([ngal_wide, num_inputs_wide])
            data_err_wide = np.zeros([ngal_wide, num_inputs_wide])
            for j, (col, errcol) in enumerate(zip(self.config.inputs_wide, self.config.input_errs_wide)):
                if self.config.convert_to_flux_wide:
                    data_wide[:, j] = mag2flux(np.array(data[col], dtype=np.float32), self.config.zero_points_wide[j])
                    data_err_wide[:, j] = magerr2fluxerr(np.array(data[errcol], dtype=np.float32), data_wide[:, j])
                else:
                    data_wide[:, j] = np.array(data[col], dtype=np.float32)
                    data_err_wide[:, j] = np.array(data[errcol], dtype=np.float32)

            # ## PUT IN THRESHOLD!
            if self.config.set_threshold_wide:
                truncation_value = self.config.thresh_value_wide
                for j in range(num_inputs_wide):
                    mask = (data_wide[:, j] < self.config.thresh_val_wide)
                    data_wide[:, j][mask] = truncation_value
                    errmask = (data_err_wide[:, j] < self.config.thresh_val_wide)
                    data_err_wide[:, j][errmask] = truncation_value

            # data_wide = data[self.config.input_errs_wide]
            data_wide_ndarray = np.array(data_wide, copy=False)
            flux_wide = data_wide_ndarray.view()
            data_err_wide_ndarray = np.array(data_err_wide, copy=False)
            flux_err_wide = data_err_wide_ndarray.view()

            cells_wide, dist_wide = self._assign_som(flux_wide, flux_err_wide, 'wide')
            if i > 1:
                widelabel = f"{label}_assignment"
            else:
                widelabel = f"{label}_wide_assignment"

            self.wide_assignment[label] = (cells_wide, dist_wide)
            self.widedict = dict(cells=cells_wide, dist=dist_wide)
            self.add_data(widelabel, self.widedict)

        tomo_bins_wide, pz_c, pc_chat, nz = self._estimate_pdf()  # *samples
        with open(self.config['tomo_bin_mask_wide_data'], 'wb') as f:
            pickle.dump(tomo_bins_wide, f)

        # Add in computation of which tomo bin each wide galaxy is mapped to
        # wide_tomo_bin_dict = self._find_wide_tomo_bins(tomo_bins_wide)
        # self.add_data("tomo_bin_mask_wide_data", wide_tomo_bin_dict)

        # self.nz = nz
        tomo_ens = qp.Ensemble(qp.interp, data=dict(xvals=self.bincents, yvals=nz))
        self.add_data('nz', tomo_ens)

        pzcdict = dict(pz_c=pz_c)
        self.add_data('pz_c', pzcdict)  # wide_data_cells_wide)

    def estimate(self,
                 spec_data,
                 balrog_data,
                 wide_data,):
        self.set_data("spec_data", spec_data)
        self.set_data("balrog_data", balrog_data)
        self.set_data("wide_data", wide_data)
        self.run()
        self.finalize()
        return


class SOMPZPzc(CatEstimator):
    """Calculate p(z|c)
    """
    name = "SOMPZPzc"
    config_options = CatEstimator.config_options.copy()
    config_options.update(inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          redshift_col=SHARED_PARAMS,
                          deep_groupname=Param(str, "photometry", msg="hdf5_groupname for deep file"),
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
                          overlap_weighted_pzc=Param(bool, False, msg="if True, use overlap_weight for p(z|c)"),
                          )
    inputs = [('spec_data', TableHandle),
              ('cell_deep_spec_data', TableHandle),]
    outputs = [('pz_c', Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band

    def run(self):
        if self.config.deep_groupname:  # pragma: no cover
            spec_data = self.get_data('spec_data')[self.config.deep_groupname]
        else:  # pragma: no cover
            spec_data = self.get_data('spec_data')
        cell_deep_spec_data = self.get_data('cell_deep_spec_data')
        self.deep_som_size = int(cell_deep_spec_data['som_size'][0])
        key = self.config.redshift_col
        zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2., self.config.zbins_max + self.config.zbins_dz, self.config.zbins_dz)
        spec_data_for_pz = pd.DataFrame({key: spec_data[key],
                                         'cell_deep': cell_deep_spec_data['cells']})
        if 'overlap_weight' in spec_data.keys(): # dtype.names:
            spec_data_for_pz['overlap_weight'] = spec_data['overlap_weight']
        all_deep_cells = np.arange(self.deep_som_size)
        pz_c = np.array(get_deep_histograms(None,  # this arg is not currently used in get_deep_histograms
                                            spec_data_for_pz,
                                            key=key,
                                            cells=all_deep_cells,
                                            overlap_weighted_pzc=self.config.overlap_weighted_pzc,
                                            bins=zbins))
        pzcdict = dict(pz_c=pz_c)
        self.add_data('pz_c', pzcdict)

    def estimate(self, spec_data, cell_deep_spec_data):
        spec_data = self.set_data('spec_data', spec_data)
        cell_deep_spec_data = self.set_data('cell_deep_spec_data', cell_deep_spec_data)
        self.run()
        self.finalize()


class SOMPZPzchat(CatEstimator):
    """Calculate p(z|chat)
    """
    name = "SOMPZPzchat"
    config_options = CatEstimator.config_options.copy()
    config_options.update(inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          redshift_col=SHARED_PARAMS,
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
                          overlap_weighted_pzc=Param(bool, False, msg="if True, use overlap_weight for p(z|c)"),
                          )
    inputs = [('spec_data', TableHandle),
              ('cell_deep_spec_data', TableHandle),
              ('cell_wide_wide_data', TableHandle),
              ('pz_c', Hdf5Handle),
              ('pc_chat', Hdf5Handle),
              ]
    outputs = [('pz_chat', Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band

    def run(self):
        spec_data = self.get_data('spec_data')
        cell_wide_wide_data = self.get_data('cell_wide_wide_data')
        cell_deep_spec_data = self.get_data('cell_deep_spec_data')
        self.wide_som_size = int(cell_wide_wide_data['som_size'][0])
        self.deep_som_size = int(cell_deep_spec_data['som_size'][0])
        pc_chat = self.get_data('pc_chat')['pc_chat'][:]
        key = self.config.redshift_col
        zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2., self.config.zbins_max + self.config.zbins_dz, self.config.zbins_dz)
        spec_data_for_pz = pd.DataFrame({key: spec_data[key],
                                         'cell_deep': cell_deep_spec_data['cells']})
        if 'overlap_weight' in spec_data.keys(): # dtype.names:
            spec_data_for_pz['overlap_weight'] = spec_data['overlap_weight']

        wide_data_for_pz = pd.DataFrame({'cell_wide': cell_wide_wide_data['cells']})

        all_wide_cells = np.arange(self.wide_som_size)
        all_deep_cells = np.arange(self.deep_som_size)

        pz_chat = np.array(histogram(wide_data_for_pz,
                                     spec_data_for_pz,
                                     key=key,
                                     pcchat=pc_chat,
                                     cells=all_wide_cells,
                                     cell_weights=np.ones(len(all_wide_cells)),
                                     deep_som_size=self.deep_som_size,
                                     overlap_weighted_pzc=self.config.overlap_weighted_pzc,
                                     bins=zbins,
                                     individual_chat=True))
        pzchatdict = dict(pz_chat=pz_chat)
        self.add_data('pz_chat', pzchatdict)

    def estimate(self, spec_data, cell_deep_spec_data, cell_wide_wide_data, pz_c, pc_chat):
        self.set_data("spec_data", spec_data)
        self.set_data("cell_deep_spec_data", cell_deep_spec_data)
        self.set_data("cell_wide_wide_data", cell_wide_wide_data)
        self.set_data("pz_c", pz_c)
        self.set_data("pc_chat", pc_chat)
        self.run()
        self.finalize()


class SOMPZPc_chat(CatEstimator):
    """Calculate p(c|chat)
    """
    name = "SOMPZPc_chat"
    config_options = CatEstimator.config_options.copy()
    config_options.update(inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          )
    inputs = [('cell_deep_balrog_data', TableHandle),
              ('cell_wide_balrog_data', TableHandle),
              ]
    outputs = [('pc_chat', Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band

    def run(self):
        cell_deep_balrog_data = self.get_data('cell_deep_balrog_data')
        cell_wide_balrog_data = self.get_data('cell_wide_balrog_data')
        self.deep_som_size = int(cell_deep_balrog_data['som_size'][0])
        self.wide_som_size = int(cell_wide_balrog_data['som_size'][0])
        pc_chat = calculate_pcchat(self.deep_som_size,
                                   self.wide_som_size,
                                   cell_deep_balrog_data['cells'],  # balrog_data['cell_deep'],#.values,
                                   cell_wide_balrog_data['cells'],  # balrog_data['cell_wide'],#.values,
                                   np.ones(len(cell_wide_balrog_data['cells'])))
        pcchatdict = dict(pc_chat=pc_chat)
        self.add_data('pc_chat', pcchatdict)

    def estimate(self, cell_deep_balrog_data, cell_wide_balrog_data):
        self.set_data('cell_deep_balrog_data', cell_deep_balrog_data)
        self.set_data('cell_wide_balrog_data', cell_wide_balrog_data)
        self.run()
        self.finalize()


class SOMPZTomobin(CatEstimator):
    """Calculate tomobin
    """
    name = "SOMPZTomobin"
    config_options = CatEstimator.config_options.copy()
    config_options.update(inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          redshift_col=SHARED_PARAMS,
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
                          overlap_weighted=Param(bool, False, msg="if True, use overlap_weight when defining tomographic bins"),
                          )
    inputs = [('spec_data', TableHandle),
              ('cell_deep_spec_data', TableHandle),
              ('cell_wide_spec_data', TableHandle),
              ('balrog_data', TableHandle),
              ('cell_deep_balrog_data', TableHandle),
              ('cell_wide_balrog_data', TableHandle),
              ('pz_c', Hdf5Handle),
              ('pc_chat', Hdf5Handle),
              ]
    outputs = [('tomo_bins_wide', Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band

    def run(self):
        spec_data = self.get_data('spec_data')
        balrog_data = self.get_data('balrog_data')
        cell_deep_spec_data = self.get_data('cell_deep_spec_data')
        cell_wide_spec_data = self.get_data('cell_wide_spec_data')
        cell_deep_balrog_data = self.get_data('cell_deep_balrog_data')
        cell_wide_balrog_data = self.get_data('cell_wide_balrog_data')
        # print('cell_wide_spec_data', cell_wide_spec_data['som_size'])
        # print('cell_deep_spec_data', cell_deep_spec_data['som_size'])
        
        self.wide_som_size = int(cell_wide_spec_data['som_size'][0])
        self.deep_som_size = int(cell_deep_spec_data['som_size'][0])
        # pc_chat = self.get_data('pc_chat')['pc_chat'][:]
        key = self.config.redshift_col
        zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2., self.config.zbins_max + self.config.zbins_dz, self.config.zbins_dz)

        pz_c = self.get_data('pz_c')['pz_c'][:]
        pc_chat = self.get_data('pc_chat')['pc_chat'][:]

        spec_data_for_pz = pd.DataFrame({key: spec_data[key],
                                         'cell_deep': cell_deep_spec_data['cells'],
                                         'cell_wide': cell_wide_spec_data['cells']})
        balrog_data_for_pz = pd.DataFrame({key: balrog_data[key],
                                         'cell_deep': cell_deep_balrog_data['cells'],
                                         'cell_wide': cell_wide_balrog_data['cells']})
        if 'overlap_weight' in balrog_data.keys(): # dtype.names:
            balrog_data_for_pz['overlap_weight'] = balrog_data['overlap_weight']
        # DES Y3-like tomographic binning
        # tomo_bins_wide_dict = define_tomo_bins_modal_spec(spec_data_for_pz,
        #                                           self.deep_som_size,
        #                                           self.wide_som_size,
        #                                           bin_edges=self.config['bin_edges'],
        #                                           key_z=key,
        #                                           key_cells_wide='cell_wide')
        # Buchs-like tomographic binning
        # tomo_bins_deep_dict = define_tomo_bins_deep(balrog_data_for_pz,
        #                                             self.deep_som_size,
        #                                             # self.deep_model['som'].shape,
        #                                             overlap_weighted=self.config.overlap_weighted,
        #                                             n_bins=len(self.config.bin_edges)-1,
        #                                             key=key,
        #                                             cell_key='cell_deep')
        tomo_bins_deep_dict = define_tomo_bins_deep_fast(
            balrog_data_for_pz,
            self.deep_som_size,
            overlap_weighted=self.config.overlap_weighted,
            n_bins=len(self.config.bin_edges) - 1,
            key=key,
            cell_key='cell_deep',
            pz_c=pz_c,
            zbins=zbins,
        )
        tomo_bins_deep = tomo_bins_wide_2d(tomo_bins_deep_dict)
        tomo_bins_deep_mapping = -1 * np.ones((self.deep_som_size, 2))
        for key in tomo_bins_deep:
            tomo_bins_deep_mapping[tomo_bins_deep[key][:, 0].astype(int), 0] = key
            tomo_bins_deep_mapping[tomo_bins_deep[key][:, 0].astype(int), 1] = tomo_bins_deep[key][:, 1]
        
        tomo_bins_wide_dict = define_tomo_bins_wide(pc_chat, tomo_bins_deep_dict)
        tomo_bins_wide = tomo_bins_wide_2d(tomo_bins_wide_dict)
        tomo_bins_mapping = -1 * np.ones((self.wide_som_size, 2))
        for key in tomo_bins_wide:
            tomo_bins_mapping[tomo_bins_wide[key][:, 0].astype(int), 0] = key
            tomo_bins_mapping[tomo_bins_wide[key][:, 0].astype(int), 1] = tomo_bins_wide[key][:, 1]
        # self.add_data('tomo_bins_deep', dict(tomo_bins_deep=tomo_bins_deep_mapping))
        self.add_data('tomo_bins_wide', dict(tomo_bins_wide=tomo_bins_mapping))

    def estimate(self, spec_data, cell_deep_spec_data, cell_wide_spec_data,
                 balrog_data, cell_deep_balrog_data, cell_wide_balrog_data, 
                 pz_c, pc_chat):
        self.set_data('spec_data', spec_data)
        self.set_data('cell_deep_spec_data', cell_deep_spec_data)
        self.set_data('cell_wide_spec_data', cell_wide_spec_data)
        self.set_data('balrog_data', balrog_data)
        self.set_data('cell_deep_balrog_data', cell_deep_balrog_data)
        self.set_data('cell_wide_balrog_data', cell_wide_balrog_data)
        self.set_data('pz_c', pz_c)
        self.set_data('pc_chat', pc_chat)
        self.run()
        self.finalize()


class SOMPZnz(CatEstimator):
    """Calculate nz
    """
    name = "SOMPZnz"
    config_options = CatEstimator.config_options.copy()
    config_options.update(inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          redshift_col=SHARED_PARAMS,
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
                          overlap_weighted_pchat=Param(bool, False, msg="if True, use overlap_weight for p(chat)"),
                          overlap_weighted_pzc=Param(bool, False, msg="if True, use overlap_weight for p(z|c)"),
                          use_bin_conditioning=Param(bool, False, msg="if True, make bin-conditioned n(z)"),
                          )
    inputs = [('spec_data', TableHandle),
              ('cell_deep_spec_data', TableHandle),
              ('cell_wide_wide_data', TableHandle),
              ('tomo_bins_wide', Hdf5Handle),
              ('pc_chat', Hdf5Handle),
              ]
    outputs = [('nz', QPHandle)]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band

    def run(self):
        spec_data = self.get_data('spec_data')
        cell_wide_wide_data = self.get_data('cell_wide_wide_data')
        cell_deep_spec_data = self.get_data('cell_deep_spec_data')
        self.wide_som_size = int(cell_wide_wide_data['som_size'][0])
        self.deep_som_size = int(cell_deep_spec_data['som_size'][0])
        tomo_bins_wide_in = self.get_data('tomo_bins_wide')['tomo_bins_wide'][:]
        tomo_bins_wide = {}
        for i in np.unique(tomo_bins_wide_in[:, 0]):
            if i < 0:
                continue
            inarr1 = np.where(tomo_bins_wide_in[:, 0] == i)[0]
            inarr2 = tomo_bins_wide_in[inarr1, 1]
            tomo_bins_wide[i] = np.array([inarr1, inarr2]).T
        pc_chat = self.get_data('pc_chat')['pc_chat'][:]
        key = self.config.redshift_col
        zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2., self.config.zbins_max + self.config.zbins_dz, self.config.zbins_dz)
        spec_data_for_pz = pd.DataFrame({key: spec_data[key],
                                         'cell_deep': cell_deep_spec_data['cells']})
        if 'overlap_weight' in spec_data.keys(): # dtype.names:
            spec_data_for_pz['overlap_weight'] = spec_data['overlap_weight']

        wide_data_for_pz = pd.DataFrame({'cell_wide': cell_wide_wide_data['cells']})
        if 'overlap_weight' in cell_wide_wide_data.keys(): # dtype.names:
            wide_data_for_pz['overlap_weight'] = cell_wide_wide_data['overlap_weight']

        all_wide_cells = np.arange(self.wide_som_size)
        all_deep_cells = np.arange(self.deep_som_size)
        if self.config.use_bin_conditioning:
            nz = np.array([nz_bin_conditioned(wide_data_for_pz, 
                                              spec_data_for_pz, 
                                              overlap_weighted_pchat=True, # intentionally hard-coded here
                                              overlap_weighted_pzc=True, # intentionally hard-coded here
                                              tomo_cells=tomo_bins_wide[i], 
                                              zbins=zbins, 
                                              pcchat = pc_chat, 
                                              cell_wide_key='cell_wide', 
                                              zkey=key) for i in range(len(tomo_bins_wide))])
        else:
            nz = redshift_distributions_wide(data=wide_data_for_pz,
                                             deep_data=spec_data_for_pz,
                                             overlap_weighted_pchat=self.config.overlap_weighted_pchat,
                                             overlap_weighted_pzc=self.config.overlap_weighted_pzc,
                                             bins=zbins,
                                             deep_som_size=self.deep_som_size,
                                             pcchat=pc_chat,
                                             tomo_bins=tomo_bins_wide,
                                             key=key,
                                             force_assignment=False,
                                             cell_key='cell_wide')
        self.bincents = 0.5 * (zbins[1:] + zbins[:-1])
        tomo_ens = qp.Ensemble(qp.interp, data=dict(xvals=self.bincents, yvals=nz))
        self.add_data('nz', tomo_ens)

    def estimate(self, spec_data, cell_deep_spec_data, cell_wide_wide_data, tomo_bins_wide, pc_chat):
        spec_data = self.set_data('spec_data', spec_data)
        cell_deep_spec_data = self.set_data('cell_deep_spec_data', cell_deep_spec_data)
        cell_wide_wide_data = self.set_data('cell_wide_wide_data', cell_wide_wide_data)
        tomo_bins_wide = self.set_data('tomo_bins_wide', tomo_bins_wide)
        pc_chat = self.set_data('pc_chat', pc_chat)
        self.run()
        self.finalize()


class SOMPZnz_fast(CatEstimator):
    """Calculate nz using precomputed p(z|c) and p(c|chat)."""
    name = "SOMPZnz_fast"
    config_options = CatEstimator.config_options.copy()
    config_options.update(inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          redshift_col=SHARED_PARAMS,
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
                          overlap_weighted_pchat=Param(bool, False, msg="if True, use overlap_weight for p(chat)"),
                          overlap_weighted_pzc=Param(bool, False, msg="expected overlap weighting used to build input p(z|c)"),
                          )
    # Keep interface close to SOMPZnz so it is easy to swap stages in YAML
    # spec_data and cell_deep_spec_data are accepted for drop-in compatibility
    # but are not required here
    inputs = [('spec_data', TableHandle),
              ('cell_deep_spec_data', TableHandle),
              ('cell_wide_wide_data', TableHandle),
              ('wide_data', TableHandle),
              ('tomo_bins_wide', Hdf5Handle),
              ('pc_chat', Hdf5Handle),
              ('pz_c', Hdf5Handle),
              ]
    outputs = [('nz', QPHandle)]

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def _parse_tomo_bins_wide(self, tomo_bins_wide_in):
        tomo_bins_wide = {}
        for i in np.unique(tomo_bins_wide_in[:, 0]):
            if i < 0:
                continue
            inds = np.where(tomo_bins_wide_in[:, 0] == i)[0]
            weights = tomo_bins_wide_in[inds, 1]
            tomo_bins_wide[int(i)] = np.array([inds, weights]).T
        return tomo_bins_wide

    def run(self):
        cell_wide_wide_data = self.get_data('cell_wide_wide_data')
        wide_cells = np.asarray(cell_wide_wide_data['cells'], dtype=np.int64)
        self.wide_som_size = int(cell_wide_wide_data['som_size'][0])

        tomo_bins_wide_in = self.get_data('tomo_bins_wide')['tomo_bins_wide'][:]
        tomo_bins_wide = self._parse_tomo_bins_wide(tomo_bins_wide_in)

        pc_chat = np.asarray(self.get_data('pc_chat')['pc_chat'][:], dtype=float)
        pz_c = np.asarray(self.get_data('pz_c')['pz_c'][:], dtype=float)
        self.deep_som_size = pz_c.shape[0]

        zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2.,
                          self.config.zbins_max + self.config.zbins_dz,
                          self.config.zbins_dz)
        dz = np.diff(zbins)

        if pc_chat.shape[0] != self.deep_som_size:
            raise ValueError(f"pc_chat deep axis ({pc_chat.shape[0]}) != pz_c deep axis ({self.deep_som_size})")
        if pc_chat.shape[1] != self.wide_som_size:
            raise ValueError(f"pc_chat wide axis ({pc_chat.shape[1]}) != wide som size ({self.wide_som_size})")

        # p(chat | sample): fraction of sample in each wide SOM cell.
        if self.config.overlap_weighted_pchat != self.config.overlap_weighted_pzc:
            raise ValueError(
                "Inconsistent overlap-weighting config in SOMPZnz_fast: "
                f"overlap_weighted_pchat={self.config.overlap_weighted_pchat} "
                f"!= overlap_weighted_pzc={self.config.overlap_weighted_pzc}. "
                "The current implementation requires overlap_weighted_pchat and overlap_weighted_pzc to be the same here."
            )

        if self.config.overlap_weighted_pchat:
            wide_data = self.get_data('wide_data')
            if 'overlap_weight' not in wide_data.keys(): # dtype.names:
                raise ValueError("overlap_weighted_pchat=True, but overlap_weight is missing from wide_data.")
            overlap_weight = np.asarray(wide_data['overlap_weight'], dtype=float)
            if overlap_weight.shape[0] != wide_cells.shape[0]:
                raise ValueError(
                    f"wide_data length ({overlap_weight.shape[0]}) must match cell_wide_wide_data length ({wide_cells.shape[0]})."
                )
            chat_counts = np.bincount(wide_cells, weights=overlap_weight, minlength=self.wide_som_size).astype(float)
        else:
            chat_counts = np.bincount(wide_cells, minlength=self.wide_som_size).astype(float)
        if chat_counts.sum() == 0:
            raise ValueError("No wide-cell assignments found; cannot compute nz.")
        p_chat_sample = chat_counts / chat_counts.sum()

        # Combine p(chat|sample), tomo-bin weights, and p(c|chat), then marginalize
        # deep-cell histograms p(z|c) to get n(z) per tomo bin.
        bin_keys = sorted(tomo_bins_wide.keys())
        nz_rows = []
        for bin_key in bin_keys:
            cells_use = tomo_bins_wide[bin_key][:, 0].astype(np.int64)
            cells_binweights = tomo_bins_wide[bin_key][:, 1].astype(float)

            p_chat_bin = np.zeros(self.wide_som_size, dtype=float)
            p_chat_bin[cells_use] = p_chat_sample[cells_use] * cells_binweights

            p_c_bin = np.sum(pc_chat * p_chat_bin[None, :], axis=1)
            hist = np.sum(pz_c * p_c_bin[:, None], axis=0)

            norm = np.sum(hist * dz)
            if norm > 0:
                hist = hist / norm
            nz_rows.append(hist)

        nz = np.array(nz_rows)
        self.bincents = 0.5 * (zbins[1:] + zbins[:-1])
        tomo_ens = qp.Ensemble(qp.interp, data=dict(xvals=self.bincents, yvals=nz))
        self.add_data('nz', tomo_ens)

    def estimate(self, spec_data, cell_deep_spec_data, cell_wide_wide_data, wide_data, tomo_bins_wide, pc_chat, pz_c):
        self.set_data('spec_data', spec_data)
        self.set_data('cell_deep_spec_data', cell_deep_spec_data)
        self.set_data('cell_wide_wide_data', cell_wide_wide_data)
        self.set_data('wide_data', wide_data)
        self.set_data('tomo_bins_wide', tomo_bins_wide)
        self.set_data('pc_chat', pc_chat)
        self.set_data('pz_c', pz_c)
        self.run()
        self.finalize()


class SOMPZEstimatorBase(CatEstimator):
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "SOMPZEstimatorBase"
    config_options = CatEstimator.config_options.copy()
    config_options.update(chunk_size=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS,
                          hdf5_groupname=SHARED_PARAMS,
                          inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          input_errs=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for deep data"),
                          zero_points=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for deep data, if needed"),
                          som_shape=Param(list, [32, 32], msg="shape for the deep som, must be a 2-element tuple"),
                          som_minerror=Param(float, 0.01, msg="floor placed on observational error on each feature in deep som"),
                          som_wrap=Param(bool, False, msg="flag to set whether the deep SOM has periodic boundary conditions"),
                          som_take_log=Param(bool, True, msg="flag to set whether to take log of inputs (i.e. for fluxes) for deep som"),
                          convert_to_flux=Param(bool, False, msg="flag for whether to convert input columns to fluxes for deep data"
                                                     "set to true if inputs are mags and to False if inputs are already fluxes"),
                          set_threshold=Param(bool, False, msg="flag for whether to replace values below a threshold with a set number"),
                          thresh_val=Param(float, 1.e-5, msg="threshold value for set_threshold for deep data"),
                          debug=Param(bool, False, msg="boolean reducing dataset size for quick debugging"))

    inputs = [('model', ModelHandle),
              ('data', TableHandle),]
    outputs = [
        ('assignment', Hdf5Handle),
    ]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band
        if len(self.config.inputs) != len(self.config.input_errs):  # pragma: no cover
            raise ValueError("Number of inputs_deep specified in inputs_deep must be equal to number of mag errors specified in input_errs_deep!")
        if len(self.config.som_shape) != 2:  # pragma: no cover
            raise ValueError(f"som_shape must be a list with two integers specifying the SOM shape, not len {len(self.config.som_shape)}")

    def _assign_som(self, flux, flux_err):
        # som_dim = self.config.som_shape[0]
        s0 = int(self.config.som_shape[0])
        s1 = int(self.config.som_shape[1])
        self.som_size = np.array([int(s0 * s1)])

        nTrain = flux.shape[0]
        som_weights = self.model['som'].weights
        hh = somfuncs.hFunc(nTrain, sigma=(30, 1))
        metric = somfuncs.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
        som = somfuncs.NoiseSOM(metric, None, None,
                                learning=hh,
                                shape=(s0, s1),
                                wrap=False, logF=True,
                                initialize=som_weights,
                                minError=0.02)
        subsamp = 1
        cells_test, dist_test = som.classify(flux[::subsamp, :], flux_err[::subsamp, :])

        return cells_test, dist_test

    def _process_chunk(self, start, end, data, first):
        """
        Run SOMPZ on a chunk of data
        """
        ngal_wide = len(data[self.config.inputs[0]])
        num_inputs_wide = len(self.config.inputs)
        data_wide = np.zeros([ngal_wide, num_inputs_wide])
        data_err_wide = np.zeros([ngal_wide, num_inputs_wide])
        for j, (col, errcol) in enumerate(zip(self.config.inputs, self.config.input_errs)):
            if self.config.convert_to_flux:
                data_wide[:, j] = mag2flux(np.array(data[col], dtype=np.float32), self.config.zero_points[j])
                data_err_wide[:, j] = magerr2fluxerr(np.array(data[errcol], dtype=np.float32), data_wide[:, j])
            else:  # pragma: no cover
                data_wide[:, j] = np.array(data[col], dtype=np.float32)
                data_err_wide[:, j] = np.array(data[errcol], dtype=np.float32)

        if self.config.set_threshold:
            truncation_value = self.config.thresh_val
            for j in range(num_inputs_wide):
                mask = (data_wide[:, j] < self.config.thresh_val)
                data_wide[:, j][mask] = truncation_value
                errmask = (data_err_wide[:, j] < self.config.thresh_val)
                data_err_wide[:, j][errmask] = truncation_value

        data_wide_ndarray = np.array(data_wide, copy=False)
        flux_wide = data_wide_ndarray.view()
        data_err_wide_ndarray = np.array(data_err_wide, copy=False)
        flux_err_wide = data_err_wide_ndarray.view()

        cells_wide, dist_wide = self._assign_som(flux_wide, flux_err_wide)
        output_chunk = dict(cells=cells_wide, dist=dist_wide)
        self._do_chunk_output(output_chunk, start, end, first)

    def _do_chunk_output(self, output_chunk, start, end, first):
        """

        Parameters
        ----------
        output_chunk
        start
        end
        first

        Returns
        -------

        """
        if first:
            self._output_handle = self.add_handle('assignment', data=output_chunk)
            self._output_handle.initialize_write(self._input_length, communicator=self.comm)
        self._output_handle.set_data(output_chunk, partial=True)
        self._output_handle.write_chunk(start, end)

    def run(self):
        self.model = None
        self.model = self.open_model(**self.config)  # None
        first = True
        if self.config.hdf5_groupname:  # pragma: no cover
            # print(self.config.hdf5_groupname)
            self.input_iterator('data')
            iter1 = self.input_iterator('data')[self.config.hdf5_groupname]
        else:
            iter1 = self.input_iterator('data')
        # iter1 = self.input_iterator('data', groupname=self.config.hdf5_groupname)
        # iter1 = self.input_iterator('data')
        self._output_handle = None
        for s, e, test_data in iter1:
            print(f"Process {self.rank} running creator on chunk {s} - {e}", flush=True)
            self._process_chunk(s, e, test_data, first)
            first = False
            gc.collect()
        if self.comm:  # pragma: no cover
            self.comm.Barrier()
        self._finalize_run()

    def estimate(self, data):
        # print('set data', data)
        self.set_data("data", data)
        self.run()
        self.finalize()
        return

    def _finalize_run(self):
        """

        Returns
        -------

        """
        tmpdict = dict(som_size=self.som_size) # add som size to assignment output # likely can reconfigure such that this is not needed
        self._output_handle.finalize_write(**tmpdict)
        # self._output_handle.finalize_write({})


class SOMPZEstimatorWide(SOMPZEstimatorBase):
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "SOMPZEstimatorWide"

    inputs = [('wide_model', ModelHandle),
              ('data', TableHandle),]

    def open_model(self, **kwargs):
        """Load the model and/or attach it to this Creator.

        Keywords
        --------
        model : object, str or ModelHandle
            Either an object with a trained model, a path pointing to a file
            that can be read to obtain the trained model, or a ``ModelHandle``
            providing access to the trained model

        Returns
        -------
        self.model : object
            The object encapsulating the trained model
        """
        model = kwargs.get("model", kwargs.get('wide_model', None))
        if model is None or model == "None":  # pragma: no cover
            self.model = None
        else:
            if isinstance(model, str):
                self.model = self.set_data("wide_model", data=None, path=model)
                self.config["model"] = model
            else:  # pragma: no cover
                if isinstance(model, ModelHandle):  # pragma: no cover
                    if model.has_path:
                        self.config["model"] = model.path
                self.model = self.set_data("wide_model", model)

        return self.model


class SOMPZEstimatorDeep(SOMPZEstimatorBase):
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "SOMPZEstimatorDeep"
    inputs = [('deep_model', ModelHandle),
              ('data', TableHandle),]

    def open_model(self, **kwargs):
        """Load the model and/or attach it to this Creator.

        Keywords
        --------
        model : object, str or ModelHandle
            Either an object with a trained model, a path pointing to a file
            that can be read to obtain the trained model, or a ``ModelHandle``
            providing access to the trained model

        Returns
        -------
        self.model : object
            The object encapsulating the trained model
        """
        model = kwargs.get("model", kwargs.get('deep_model', None))
        if model is None or model == "None":  # pragma: no cover
            self.model = None
        else:
            if isinstance(model, str):
                self.model = self.set_data("deep_model", data=None, path=model)
                self.config["model"] = model
            else:  # pragma: no cover
                if isinstance(model, ModelHandle):  # pragma: no cover
                    if model.has_path:
                        self.config["model"] = model.path
                self.model = self.set_data("deep_model", model)

        return self.model
