#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
from rail.estimation.algos.sompz import SOMPZInformer, SOMPZEstimator
from rail.evaluation.single_evaluator import SingleEvaluator
from rail.estimation.algos.true_nz import TrueNZHistogrammer

from rail.core.stage import RailStage, RailPipeline
from rail.core import common_params

import ceci

som_params = dict(
    convert_to_flux_deep=False, convert_to_flux_wide=False, 
    set_threshold_deep=True, thresh_val_deep=1.e-5,
    som_shape_deep=[25, 25],
    som_shape_wide=[25, 25], som_minerror_wide=0.005,
    som_take_log_wide=False, som_wrap_wide=False,
)


class SOMPZPipeline(RailPipeline):

    default_input_dict = dict(
        train_deep_data = 'dummy.hdf5',
        train_wide_data = 'dummy.hdf5',
        test_spec_data = 'dummy.hdf5',
        test_balrog_data = 'dummy.hdf5',
        test_wide_data = 'dummy.hdf5',
        truth = 'dummy.hdf5',
    )

    def __init__(self, bin_edges=[0.0, 0.405, 0.665, 0.96, 2.0]):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        deep_bands = common_params.SHARED_PARAMS['bands'].copy()
        deep_errs = common_params.SHARED_PARAMS['err_bands'].copy()

        wide_bands = common_params.SHARED_PARAMS['bands'].copy()
        wide_errs = common_params.SHARED_PARAMS['err_bands'].copy()

        deep_groupname = common_params.SHARED_PARAMS['hdf5_groupname']
        wide_groupname = common_params.SHARED_PARAMS['hdf5_groupname']
        spec_groupname = common_params.SHARED_PARAMS['hdf5_groupname']
        balrog_groupname = common_params.SHARED_PARAMS['hdf5_groupname']
        
        refband_deep=deep_bands[3]
        refband_wide=wide_bands[3]        
        zero_pts = [30. for _band in deep_bands]

        self.inform_sompz = SOMPZInformer.build(
            aliases=dict(
                input_deep_data='train_deep_data',
                input_wide_data='train_wide_data',
            ),
            deep_groupname=deep_groupname, 
            wide_groupname=wide_groupname,
            inputs_deep=deep_bands,
            input_errs_deep=deep_errs,
            zero_points_deep=zero_pts, 
            inputs_wide=wide_bands,
            input_errs_wide=wide_errs,
            **som_params,
        )

        self.estimate_sompz = SOMPZEstimator.build(
            aliases=dict(
                spec_data='test_spec_data',
                balrog_data='test_balrog_data',
                wide_data='test_wide_data',
            ),
            connections=dict(
                model=self.inform_sompz.io.model,
            ),
            bin_edges=bin_edges
            spec_groupname=spec_groupname,
            balrog_groupname=balrog_groupname,
            wide_groupname=wide_groupname,
            deep_bands=deep_bands,
            err_deep_bands=deep_errs,  
            inputs_deep=deep_bands,
            input_errs_deep=deep_errs,
            inputs_wide=wide_bands,
            input_errs_wide=wide_errs,
            ref_band_deep=refband_deep,
            wide_bands=wide_bands,
            err_wide_bands=wide_errs,
            ref_band_wide=refband_wide,
            **som_params,
        )

        n_tomo_bins = len(bin_edges) - 1
        for ibin in range(n_tomo_bins):
            true_nz = TrueNZHistogrammer.make_and_connect(
                name=f"true_nz_sompz_bin{ibin}",
                connections=dict(
                    tomography_bins=self.estimate_sompz.io.tomo_bin_assignment,
                ),
                selected_bin=ibin,
                aliases=dict(input='truth'),
            )
            self.add_stage(true_nz)
