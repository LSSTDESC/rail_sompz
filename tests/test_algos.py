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
from rail.utils.path_utils import RAILDIR
from rail.utils.testing_utils import one_algo
from rail.estimation.algos import sompz
from rail.sompz.utils import RAIL_SOMPZ_DIR

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
def test_sompz_train(ntarray):
    """
    # first, train with two broad types
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'dz': 0.01, 'hdf5_groupname': 'photometry',
                         'nt_array': ntarray, 'type_file': 'tmp_broad_types.hdf5',
                         'model': 'testmodel_sompz.pkl'}
    if len(ntarray) == 2:
        broad_types = np.random.randint(2, size=100)
    else:
        broad_types = np.zeros(100, dtype=int)
    typedict = dict(types=broad_types)
    tables_io.write(typedict, "tmp_broad_types.hdf5")
    train_algo = sompz.SOMPZInformer
    DS.clear()
    training_data = DS.read_file('training_data', TableHandle, traindata)
    train_stage = train_algo.make_stage(**train_config_dict)
    train_stage.inform(training_data)
    expected_keys = ['fo_arr', 'kt_arr', 'zo_arr', 'km_arr', 'a_arr', 'mo', 'nt_array']
    with open("testmodel_sompz.pkl", "rb") as f:
        tmpmodel = pickle.load(f)
    for key in expected_keys:
        assert key in tmpmodel.keys()
    os.remove("tmp_broad_types.hdf5")
    """


@pytest.mark.parametrize(
    "inputdata, groupname",
    [
        (parquetdata, ""),
        (validdata, "photometry")
    ]
)
def test_sompz(inputdata, groupname):
    """
    train_config_dict = {}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0,
                         'dz': 0.01,
                         'nzbins': 301,
                         'data_path': None,
                         'columns_file': os.path.join(RAIL_SOMPZ_DIR, "rail/examples_data/estimation_data/configs/test_sompz.columns"),
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
                         'model': 'testmodel_sompz.pkl'}
    zb_expected = np.array([0.16, 0.12, 0.14, 0.14, 0.06, 0.14, 0.12, 0.14, 0.06, 0.16])
    train_algo = None
    pz_algo = sompz.SOMPZEstimator
    results, rerun_results, rerun3_results = one_algo("SOMPZ", train_algo, pz_algo, train_config_dict, estim_config_dict)
    assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()
    """
