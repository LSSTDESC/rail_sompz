"""
Port of SOMPZ
"""
import pdb
import os
import numpy as np
# import sys
import qp
from ceci.config import StageParameter as Param
from rail.core.data import TableHandle, ModelHandle, FitsHandle, QPHandle, Hdf5Handle
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.core.utils import RAILDIR
import rail.estimation.algos.som as somfuncs
from rail.core.common_params import SHARED_PARAMS

# import astropy.io.fits as fits  # TODO handle file i/o with rail
import pandas as pd
import matplotlib.pyplot as plt


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

    if len(interpolate_kwargs) > 0:
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
                if overlap_weighted_pzc:
                    # print("WARNING: You are using a deprecated point estimate Z. No overlap weighting enabled.
                    # You're on your own now.")#suppress
                    weights = df[overlap_key].values
                else:
                    weights = np.ones(len(z))
                hist = np.histogram(z, bins, weights=weights, density=True)[
                    0]  # make weighted histogram by overlap weights
                populated_cells.append([ci, c])
            elif type(key) is list:
                # use full p(z)
                assert (bins is not None)
                # ##histogram_from_fullpz CURRENTLY UNDEFINED!
                hist = histogram_from_fullpz(df, key, overlap_weighted=overlap_weighted_pzc, bin_edges=bins)
            hists.append(hist)
        except KeyError as e:
            missing_cells.append([ci, c])
            hists.append(np.zeros(len(bins) - 1))
    hists = np.array(hists)

    if len(interpolate_kwargs) > 0:
        # print('Interpolating {0} missing histograms'.format(len(missing_cells)))
        missing_cells = np.array(missing_cells)
        populated_cells = np.array(populated_cells)
        hist_conds = np.isin(cells, populated_cells[:, 1]) & np.all(np.isfinite(hists), axis=1)
        for ci, c in missing_cells:
            if c not in cells_keep:
                # don't worry about interpolating cells we won't use anyways
                continue

            central_index = np.zeros(len(deep_map_shape), dtype=int)
            # unravel_index(c, deep_map_shape, central_index)  # fills central_index
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

def histogram_from_fullpz(df, key, overlap_weighted, bin_edges, full_pz_end=6.00, full_pz_npts=601):
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

    # response weight normalized p(z)
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
    if len(tomo_bins) == 0:
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
            if len(cells_conds) == 0:
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
    if overlap_weighted:
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


def bin_assignment_spec(spec_data, deep_som_size, wide_som_size, bin_edges,
                        key_z='Z', key_cells_wide='cell_wide_unsheared'):
    # assign gals in redshift sample to bins
    xlabels = []
    nbins = len(bin_edges) - 1
    for ii in range(nbins):
        xlabels.append(ii)
    spec_data['tomo_bin'] = pd.cut(spec_data[key_z], bin_edges, labels=xlabels)

    ncells_with_spec_data = len(np.unique(spec_data[key_cells_wide].values))
    cell_bin_assignment = np.ones(wide_som_size, dtype=int) * -1
    cells_with_spec_data = np.unique(spec_data[key_cells_wide].values)

    groupby_obj_value_counts = spec_data.groupby(key_cells_wide)['tomo_bin'].value_counts()

    for c in cells_with_spec_data:
        bin_assignment = groupby_obj_value_counts.loc[c].index[0]
        cell_bin_assignment[c] = bin_assignment

    # reformat bins into dict
    tomo_bins_wide = {}
    nbins = len(bin_edges) - 1
    for i in range(nbins):
        tomo_bins_wide[i] = np.where(cell_bin_assignment == i)[0]

    return tomo_bins_wide


def tomo_bins_wide_2d(tomo_bins_wide_dict):
    tomo_bins_wide = tomo_bins_wide_dict.copy()
    for k in tomo_bins_wide:
        if tomo_bins_wide[k].ndim == 1:
            tomo_bins_wide[k] = np.column_stack((tomo_bins_wide[k], np.ones(len(tomo_bins_wide[k]))))
        renorm = 1. / np.average(tomo_bins_wide[k][:, 1])
        tomo_bins_wide[k][:, 1] *= renorm  # renormalize so the mean weight is 1; important for bin conditioning
    return tomo_bins_wide


def plot_nz(hists, zbins, outfile, xlimits=(0, 2), ylimits=(0, 3.25)):
    plt.figure(figsize=(16., 9.))
    for i in range(len(hists)):
        plt.plot((zbins[1:] + zbins[:-1]) / 2., hists[i], label='bin ' + str(i))
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$p(z)$')
    plt.legend()
    plt.title('n(z)')
    plt.savefig(outfile)
    plt.close()


class SOMPZInformer(CatInformer):
    """Inform stage for SOMPZEstimator
    """
    name = "SOMPZInformer"
    config_options = CatInformer.config_options.copy()
    config_options.update(redshift_col=SHARED_PARAMS,
                          deep_groupname=Param(str, "photometry", msg="hdf5_groupname for deep data"),
                          wide_groupname=Param(str, "photometry", msg="hdf5_groupname for wide data"),
                          inputs_deep=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          input_errs_deep=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for deep data"),
                          inputs_wide=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for wide data"),
                          input_errs_wide=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for wide data"),
                          zero_points_deep=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for deep data, if needed"),
                          zero_points_wide=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for wide data, if needed"),
                          som_shape_deep=Param(tuple, (32, 32), msg="shape for the deep som, must be a 2-element tuple"),
                          som_shape_wide=Param(tuple, (32, 32), msg="shape for the wide som, must be a 2-element tuple"),
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
                          thresh_val_wide=Param(float, 1.e-5, msg="threshold value for set_threshold for wide data"))

    # inputs = [('input_spec_data', TableHandle),
    #          ('input_deep_data', TableHandle),
    #          ('input_wide_data', TableHandle),
    #          ]

    inputs = [('input_deep_data', TableHandle),
              ('input_wide_data', TableHandle),
              ]

    # outputs = [('model_som_deep', ModelHandle),
    #            ('model_som_wide', ModelHandle)]
    # ## outputs = [('model', ModelHandle)]

    def run(self):

        # note: hdf5_groupname is a SHARED_PARAM defined in the parent class!
        if self.config.deep_groupname:
            deep_data = self.get_data('input_deep_data')[self.config.deep_groupname]
        else:  # pragma: no cover
            # DEAL with hdf5_groupname stuff later, just assume it's in the top level for now!
            deep_data = self.get_data('input_deep_data')
        if self.config.wide_groupname:
            wide_data = self.get_data('input_wide_data')[self.config.wide_groupname]
        else:  # pragma: no cover
            # DEAL with hdf5_groupname stuff later, just assume it's in the top level for now!
            wide_data = self.get_data('input_wide_data')            
        # spec_data = self.get_data('input_spec_data')

        num_inputs_deep = len(self.config.inputs_deep)
        num_inputs_wide = len(self.config.inputs_wide)
        ngal_deep = len(deep_data[self.config.inputs_deep[0]])
        ngal_wide = len(wide_data[self.config.inputs_wide[0]])
        print(f"{ngal_deep} galaxies in deep sample")
        print(f"{ngal_wide} galaxies in wide sample")

        deep_input = np.zeros([ngal_deep, num_inputs_deep])
        deep_errs = np.zeros([ngal_deep, num_inputs_deep])
        wide_input = np.zeros([ngal_wide, num_inputs_wide])
        wide_errs = np.zeros([ngal_wide, num_inputs_wide])

        # assemble deep data
        for i, (col, errcol) in enumerate(zip(self.config.inputs_deep, self.config.input_errs_deep)):
            if self.config.convert_to_flux_deep:
                deep_input[:, i] = mag2flux(deep_data[col], self.config.zero_points_deep[i])
                deep_errs[:, i] = magerr2fluxerr(deep_data[errcol], deep_input[:, i])
            else:
                deep_input[:, i] = deep_data[col]
                deep_errs[:, i] = deep_data[errcol]

        # assemble wide data
        for i, (col, errcol) in enumerate(zip(self.config.inputs_wide, self.config.input_errs_wide)):
            if self.config.convert_to_flux_wide:
                wide_input[:, i] = mag2flux(wide_data[col], self.config.zero_points_deep[i])
                wide_errs[:, i] = magerr2fluxerr(wide_data[errcol], wide_input[:, i])
            else:
                wide_input[:, i] = wide_data[col]
                wide_errs[:, i] = wide_data[errcol]

        # put a temporary threshold bit in. TODO fix this up later...
        if self.config.set_threshold_deep:
            truncation_value = 1e-2
            for i in range(num_inputs_deep):
                mask = (deep_input[:, i] < self.config.thresh_val_deep)
                deep_input[:, i][mask] = truncation_value
                errmask = (deep_errs[:, i] < self.config.thresh_val_deep)
                deep_errs[:, i][errmask] = truncation_value

        if self.config.set_threshold_wide:
            truncation_value = 1e-2
            for i in range(num_inputs_wide):
                mask = (wide_input[:, i] < self.config.thresh_val_wide)
                wide_input[:, i][mask] = truncation_value
                errmask = (wide_errs[:, i] < self.config.thresh_val_wide)
                wide_errs[:, i][errmask] = truncation_value

        sommetric = somfuncs.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
        learn_func = somfuncs.hFunc(ngal_deep, sigma=(30, 1))

        print(f"Training deep SOM of shape {self.config.som_shape_deep}...")
        deep_som = somfuncs.NoiseSOM(sommetric, deep_input, deep_errs, learn_func,
                                     shape=self.config.som_shape_deep, minError=self.config.som_minerror_deep,
                                     wrap=self.config.som_wrap_deep, logF=self.config.som_take_log_deep)
        print(f"Training wide SOM of shape {self.config.som_shape_wide}...")
        learn_func = somfuncs.hFunc(ngal_wide, sigma=(30, 1))
        wide_som = somfuncs.NoiseSOM(sommetric, wide_input, wide_errs, learn_func,
                                     shape=self.config.som_shape_wide, minError=self.config.som_minerror_wide,
                                     wrap=self.config.som_wrap_wide, logF=self.config.som_take_log_wide)

        model = dict(deep_som=deep_som, wide_som=wide_som, deep_columns=self.config.inputs_deep,
                     deep_err_columns=self.config.input_errs_deep, wide_columns=self.config.inputs_wide,
                     wide_err_columns=self.config.input_errs_wide)
        self.model = model

        self.add_data('model', self.model)

    def inform(self, input_deep_data, input_wide_data):
        # self.add_data('input_spec_data', input_spec_data)
        self.set_data('input_deep_data', input_deep_data)
        self.set_data('input_wide_data', input_wide_data)

        self.run()
        self.finalize()

        return self.model


class SOMPZEstimator(CatEstimator):
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "SOMPZEstimator"
    config_options = CatEstimator.config_options.copy()
    config_options.update(redshift_col=SHARED_PARAMS,
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
#                          data_path=Param(str, "directory", msg="directory for output files"),
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
                          som_shape_deep=Param(tuple, (32, 32), msg="shape for the deep som, must be a 2-element tuple"),
                          som_shape_wide=Param(tuple, (32, 32), msg="shape for the wide som, must be a 2-element tuple"),
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
                          debug=Param(bool, False, msg="boolean reducing dataset size for quick debuggin"))

    inputs = [('model', ModelHandle),
              ('spec_data', TableHandle),
              ('balrog_data', TableHandle),
              ('wide_data', TableHandle), ]
    outputs = [('nz', QPHandle),
               ('spec_data_deep_assignment', Hdf5Handle),
               ('spec_data_wide_assignment', Hdf5Handle),               
               ('balrog_data_deep_assignment', Hdf5Handle),
               ('balrog_data_wide_assignment', Hdf5Handle),               
               ('wide_data_assignment', Hdf5Handle),
               ('pz_c', Hdf5Handle),
               ('pz_chat', Hdf5Handle),
               ('pc_chat', Hdf5Handle),
               ('tomo_bin_mask_wide_data', Hdf5Handle),
               ]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)

        '''
        datapath = self.config["data_path"]
        if datapath is None or datapath == "None":
            tmpdatapath = os.path.join(RAILDIR, "rail/examples_data/estimation_data/data")
            os.environ["SOMPZDATAPATH"] = tmpdatapath
            self.data_path = tmpdatapath
        else:  # pragma: no cover
            self.data_path = datapath
            os.environ["SOMPZDATAPATH"] = self.data_path
        if not os.path.exists(self.data_path):  # pragma: no cover
            raise FileNotFoundError("SOMPZDATAPATH " + self.data_path + " does not exist! Check value of data_path in config file!")
        '''
        
        # check on bands, errs, and prior band
        if len(self.config.inputs_deep) != len(self.config.input_errs_deep):  # pragma: no cover
            raise ValueError("Number of inputs_deep specified in inputs_deep must be equal to number of mag errors specified in input_errs_deep!")
#        if self.config.ref_band_deep not in self.config.inputs_deep:  # pragma: no cover
#            raise ValueError(f"reference band not found in inputs_deep specified in inputs_deep: {str(self.config.inputs_deep)}")

        if len(self.config.inputs_wide) != len(self.config.input_errs_wide):  # pragma: no cover
            raise ValueError("Number of inputs_wide specified in inputs_wide must be equal to number of mag errors specified in input_errs_wide!")
#        if self.config.ref_band_wide not in self.config.inputs_wide:  # pragma: no cover
#            raise ValueError(f"reference band not found in inputs_wide specified in inputs_wide: {str(self.config.inputs_wide)}")

        self.model = self.open_model(**self.config)  # None
        print('initialized model', self.model)

    def _assign_som(self, flux, flux_err, somstr):
        if somstr == 'deep':
            som_dim = self.config.som_shape_deep[0]
        elif somstr == 'wide':
            som_dim = self.config.som_shape_wide[0]

        # output_path = './'  # TODO make kwarg
        nTrain = flux.shape[0]
        # som_weights = np.load(infile_som, allow_pickle=True)
        som_weights = self.model[somstr + '_som'].weights
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
        # TODO: improve file i/o
        # output_path = './'
        deep_som_size = np.product(self.model['deep_som'].shape)
        wide_som_size = np.product(self.model['wide_som'].shape)

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
            
        if self.config.debug:
            spec_data = spec_data[:2000]
        #spec_data = self.get_data('spec_data')
        #balrog_data = self.get_data('balrog_data')
        #wide_data = self.get_data('wide_data')

        cell_deep_spec_data = self.deep_assignment['spec_data'][0]
        cell_wide_spec_data = self.wide_assignment['spec_data'][0]
        #pdb.set_trace()
        spec_data_for_pz = pd.DataFrame({key: spec_data[key],
                                         'cell_deep': cell_deep_spec_data,
                                         'cell_wide': cell_wide_spec_data})

        # compute p(z|c), redshift histograms of deep SOM cells
        pz_c = np.array(get_deep_histograms(None,  # this arg is not currently used in get_deep_histograms
                                            spec_data_for_pz,
                                            key=key,
                                            cells=all_deep_cells,
                                            overlap_weighted_pzc=False,
                                            bins=zbins))
        # compute p(c|chat,etc.), the deep-wide transfer function
        pc_chat = calculate_pcchat(deep_som_size,
                                   wide_som_size,
                                   self.deep_assignment['balrog_data'][0],  # balrog_data['cell_deep'],#.values,
                                   self.wide_assignment['balrog_data'][0],  # balrog_data['cell_wide'],#.values,
                                   np.ones(len(self.deep_assignment['balrog_data'][0])))
        pcchatdict = dict(pc_chat=pc_chat)
        self.add_data('pc_chat', pcchatdict)
        # use to write pc_chat out to file, leave in temporarily for cross checks
        # outfile = os.path.join(output_path, 'pcchat.npy')
        # np.savez(outfile, pc_chat=pc_chat)

        # compute p(chat), occupation in wide SOM cells
        all_wide_cells = np.arange(wide_som_size)
        cell_wide_wide_data = self.wide_assignment['wide_data'][0]
        wide_data_for_pz = pd.DataFrame({'cell_wide': cell_wide_wide_data})

        # compute p(z|chat) \propto sum_c p(z|c) p(c|chat)        
        pz_chat = np.array(histogram(wide_data_for_pz,
                                     spec_data_for_pz,
                                     key=key,
                                     pcchat=pc_chat,
                                     cells=all_wide_cells,
                                     cell_weights=np.ones(len(all_wide_cells)),
                                     deep_som_size=deep_som_size,
                                     overlap_weighted_pzc=False,
                                     bins=zbins,
                                     individual_chat=True))
        # note: used to write out pz_chat to np, leave in temporarily for cross-checks
        # outfile = os.path.join(output_path, 'pzchat.npy')
        # np.savez(outfile, pz_chat=pz_chat)
        pzchatdict = dict(pz_chat=pz_chat)
        self.add_data('pz_chat', pzchatdict)


        # assign sample to tomographic bins
        # bin_edges = [0.0, 0.405, 0.665, 0.96, 2.0] # this is now a config input
        # n_bins = len(self.config.bin_edges) - 1
        tomo_bins_wide_dict = bin_assignment_spec(spec_data_for_pz,
                                                  deep_som_size,
                                                  wide_som_size,
                                                  bin_edges=self.config.bin_edges,
                                                  key_z=key,
                                                  key_cells_wide='cell_wide')
        tomo_bins_wide = tomo_bins_wide_2d(tomo_bins_wide_dict)

        # np.savez("tmp_tomo_dict2d.npz", tomo_bins_wide)
        
        # compute number of galaxies per tomographic bin (diagnostic info)
        # cell_occupation_info = wide_data_for_pz.groupby('cell_wide')['cell_wide'].count()
        # bin_occupation_info = {'bin' + str(i) : np.sum(cell_occupation_info.loc[tomo_bins_wide_dict[i]].values) for i in range(n_bins)}
        # print(bin_occupation_info)

        # calculate n(z)
        nz = redshift_distributions_wide(data=wide_data_for_pz,
                                         deep_data=spec_data_for_pz,
                                         overlap_weighted_pchat=False,
                                         overlap_weighted_pzc=False,
                                         bins=zbins,
                                         deep_som_size=deep_som_size,
                                         pcchat=pc_chat,
                                         tomo_bins=tomo_bins_wide,
                                         key=key,
                                         force_assignment=False,
                                         cell_key='cell_wide')

        '''
        model_update = dict(pz_c=pz_c, pc_chat=pc_chat, pchat=p_chat,
                            pz_chat=pz_chat)
        self.model = self.model.update(model_update)
        '''
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
        wide_assign = self.widedict['cells']
        #print(tomo_bins_wide)

        #nbins = len(self.config.bin_edges)-1
        #ngal = len(wide_assign)
        #tomo_mask = np.zeros(ngal, dtype=int)
        tmp_cells = np.concatenate([tomo_bins_wide[nbin][:,0].astype(np.int32) for nbin in tomo_bins_wide])
        tmp_weights = np.concatenate([tomo_bins_wide[nbin][:,1] for nbin in tomo_bins_wide])
        tmp_bins = np.concatenate([(np.ones(len(tomo_bins_wide[nbin][:,0])) * nbin).astype(int) for nbin in tomo_bins_wide])
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
        assert False

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

        iterator = self.input_iterator("wide_data")
        first = True
        self._initialize_run() # TODO implement
        self._output_handle = None # TODO consider handle for dict to store all outputs
        for s, e, data_chunk in iterator:
            if self.rank == 0:
                print(f"Process {self.rank} running estimator on chunk {s} - {e}")
            self._process_chunk(s, e, data_chunk, first)
            first = False
            gc.collect()

        print('You need to do spec_data and balrog_data')
        self._finalize_run()
        assert False,'below this line is code that needs to be updated'
        
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
            if i <= 1:
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
                outlabel = f"{label}_deep_assignment"
                self.add_data(outlabel, tmpdict)
            else:
                cells_deep, dist_deep = None, None

            self.wide_assignment[label] = (cells_wide, dist_wide)
            if i > 1:
                widelabel = f"{label}_assignment"
            else:
                widelabel = f"{label}_wide_assignment"
            self.widedict = dict(cells=cells_wide, dist=dist_wide)
            self.add_data(widelabel, self.widedict)

            # ## save cells_deep, dist_deep, cells_wide, dist_wide to disk
            '''
            outfile = os.path.join(output_path, label +  '_wide.npz')
            np.savez(outfile, cells=cells_wide, dist=dist_wide)

            outfile = os.path.join(output_path, label + '_incl_cells.h5')
            print('write ' + outfile)

            names = [name for name in data.colnames if len(data[name].shape) <= 1]
            df_out = data[names].to_pandas()
            df_out.to_hdf(outfile, key=label)
            #fits.writeto(outfile, data.as_array(), overwrite=True)
            '''
        tomo_bins_wide, pz_c, pc_chat, nz = self._estimate_pdf()  # *samples

        # Add in computation of which tomo bin each wide galaxy is mapped to
        wide_tomo_bin_dict = self._find_wide_tomo_bins(tomo_bins_wide)
        self.add_data("tomo_bin_mask_wide_data", wide_tomo_bin_dict)

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
        # pdb.set_trace()
        self.finalize()

        #output = {
	#	'nz': self.get_handle("nz"),
	#	'spec_data_deep_assignment': self.get_handle("spec_data_deep_assignment"),
	#	'balrog_data_deep_assignment': self.get_handle("balrog_data_deep_assignment"),
	#	'wide_data_assignment': self.get_handle("wide_data_assignment"),
	#	'pz_c': self.get_handle("pz_c"),
	#	'pz_chat': self.get_handle("pz_chat"),
	#	'pc_chat': self.get_handle("pc_chat"),
	#}
        #return output
        return
