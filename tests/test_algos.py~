import numpy as np
import os
import sys
import glob
import pickle
import pytest
import yaml
import tables_io
from rail.core.stage import RailStage
from rail.core.data import DataStore, TableHandle
from rail.core.utils import RAILDIR
from rail.core.algo_utils import one_algo
from rail.estimation.algos import bpz_lite
from rail.bpz.utils import RAIL_BPZ_DIR

import scipy.special
sci_ver_str = scipy.__version__.split('.')

parquetdata = "./tests/validation_10gal.pq"
traindata = os.path.join(RAILDIR, 'rail/examples_data/testdata/training_100gal.hdf5')
validdata = os.path.join(RAILDIR, 'rail/examples_data/testdata/validation_10gal.hdf5')

DS = RailStage.data_store
DS.__class__.allow_overwrite = True


@pytest.mark.parametrize(
    "ntarray",
    [[8], [4, 4]]
)
def test_bpz_train(ntarray):
    # first, train with two broad types
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'dz': 0.01, 'hdf5_groupname': 'photometry',
                         'nt_array': ntarray, 'type_file': 'tmp_broad_types.hdf5',
                         'model': 'testmodel_bpz.pkl'}
    if len(ntarray) == 2:
        broad_types = np.random.randint(2, size=100)
    else:
        broad_types = np.zeros(100, dtype=int)
    typedict = dict(types=broad_types)
    tables_io.write(typedict, "tmp_broad_types.hdf5")
    train_algo = bpz_lite.BPZliteInformer
    DS.clear()
    training_data = DS.read_file('training_data', TableHandle, traindata)
    train_stage = train_algo.make_stage(**train_config_dict)
    train_stage.inform(training_data)
    expected_keys = ['fo_arr', 'kt_arr', 'zo_arr', 'km_arr', 'a_arr', 'mo', 'nt_array']
    with open("testmodel_bpz.pkl", "rb") as f:
        tmpmodel = pickle.load(f)
    for key in expected_keys:
        assert key in tmpmodel.keys()
    os.remove("tmp_broad_types.hdf5")


def test_bpz_lite():
    train_config_dict = {}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0,
                         'dz': 0.01,
                         'nzbins': 301,
                         'data_path': None,
                         'columns_file': os.path.join(RAIL_BPZ_DIR, "rail/examples_data/estimation_data/configs/test_bpz.columns"),
                         'spectra_file': "CWWSB4.list",
                         'madau_flag': 'no',
                         'no_prior': False,
                         'ref_band': 'mag_i_lsst',
                         'prior_file': 'hdfn_gen',
                         'p_min': 0.005,
                         'gauss_kernel': 0.0,
                         'zp_errors': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                         'mag_err_min': 0.005,
                         'hdf5_groupname': 'photometry',
                         'nt_array': [8],
                         'model': 'testmodel_bpz.pkl'}
    zb_expected = np.array([0.16, 0.12, 0.14, 0.14, 0.06, 0.14, 0.12, 0.14, 0.06, 0.16])
    train_algo = None
    pz_algo = bpz_lite.BPZliteEstimator
    results, rerun_results, rerun3_results = one_algo("BPZ_lite", train_algo, pz_algo, train_config_dict, estim_config_dict)
    assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()

@pytest.mark.parametrize(
    "inputdata, groupname",
    [
        (parquetdata, ""),
        (validdata, "photometry")
        ]
)
def test_bpz_wHDFN_prior(inputdata, groupname):
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0,
                         'dz': 0.01,
                         'nzbins': 301,
                         'data_path': None,
                         'columns_file': os.path.join(RAIL_BPZ_DIR, "rail/examples_data/estimation_data/configs/test_bpz.columns"),
                         'spectra_file': "CWWSB4.list",
                         'madau_flag': 'no',
                         'ref_band': 'mag_i_lsst',
                         'prior_file': 'flat',
                         'p_min': 0.005,
                         'gauss_kernel': 0.1,
                         'zp_errors': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                         'mag_err_min': 0.005,
                         'hdf5_groupname': groupname,
                         'nt_array': [1, 2, 5],
                         'model': os.path.join(RAILDIR, 'rail/examples_data/estimation_data/data/CWW_HDFN_prior.pkl')}
    zb_expected = np.array([0.18, 2.88, 0.12, 0.15, 2.97, 2.78, 0.11, 0.19,
                            2.98, 2.92])

    validation_data = DS.read_file('validation_data', TableHandle, inputdata)
    pz = bpz_lite.BPZliteEstimator.make_stage(name='bpz_hdfn', **estim_config_dict)
    results = pz.estimate(validation_data)
    assert np.isclose(results.data.ancil['zmode'], zb_expected).all()
    DS.clear()
    os.remove(pz.get_output(pz.get_aliased_tag('output'), final_name=True))


def test_bpz_lite_wkernel_flatprior():
    train_config_dict = {}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0,
                         'dz': 0.01,
                         'nzbins': 301,
                         'data_path': None,
                         'columns_file': os.path.join(RAIL_BPZ_DIR, "rail/examples_data/estimation_data/configs/test_bpz.columns"),
                         'spectra_file': "CWWSB4.list",
                         'madau_flag': 'no',
                         'ref_band': 'mag_i_lsst',
                         'prior_file': 'flat',
                         'p_min': 0.005,
                         'gauss_kernel': 0.1,
                         'zp_errors': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                         'mag_err_min': 0.005,
                         'hdf5_groupname': 'photometry'}
    # zb_expected = np.array([0.18, 2.88, 0.12, 0.15, 2.97, 2.78, 0.11, 0.19,
    #                         2.98, 2.92])
    train_algo = None
    pz_algo = bpz_lite.BPZliteEstimator
    results, rerun_results, rerun3_results = one_algo("BPZ_lite", train_algo, pz_algo, train_config_dict, estim_config_dict)
    # assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()
