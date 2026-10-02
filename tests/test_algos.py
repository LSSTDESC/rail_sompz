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
from rail.estimation.algos.som import parallel_dsq

import scipy.special
sci_ver_str = scipy.__version__.split('.')

parquetdata = "./tests/validation_10gal.pq"
traindata = os.path.join(RAILDIR, 'rail/examples_data/testdata/training_100gal.hdf5')
validdata = os.path.join(RAILDIR, 'rail/examples_data/testdata/validation_10gal.hdf5')


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



# This is the old parallel_dsq code
def old_bottleneck(w, vnS):  # pragma: no cover
            # dn: see Eqn A6 of Sanchez+2020. Appears as asinh nu_{cb}
            dn = np.arcsinh(vnS)
            # numerator: see Eqn A6 of Sanchez+2020. Appears as asinh nu_{cb} + w_{ib} log 2 nu_{cb}
            numerator = dn + w * np.log(2 * vnS)
            return numerator, dn

def old_parallel_dsq(vn, s, w, df, h, sPenalty):
            # vnS is the re-scaled S/N of the cells, shape=(nS,nCells,nTargets,nFeatures)
            # vnS: see the paragraph containing equation A7 of Sanchez+2020
            vnS = s*vn
            numerator, dn = old_bottleneck(w, vnS)

            # dn is the asinh of the cell S/N values
            ####
            if np.any(np.isinf(numerator)):  # pragma: no cover
                #pdb.set_trace()
                print("inf numerator at: ", np.where(np.isinf(numerator)))
                print(np.any(np.isinf(w)),
                      np.any(np.isinf(vnS)),
                      np.any(np.isinf(dn)),
                      np.any(vnS <= 0))
            if np.any(np.isnan(numerator)):  # pragma: no cover
                #pdb.set_trace()
                print("nan numerator at: ", np.where(np.isnan(numerator)))
                print(f"found nan in: w={np.any(np.isnan(w))}, vnS={np.any(np.isnan(vnS))}, dn={np.any(np.isnan(dn))}; vnS <= 0={np.any(vnS <= 0)}")

            dn = numerator / (1 + w)
            d = (dn - df) * h
            dsq0 = np.sum(d * d, axis=3)  # Sum distance over features
            # Now add penalty for the scaling factor
            dsq0 +=  sPenalty
            # Take minimum distance of all scaling factors
            return np.min(dsq0, axis=0)

def test_new_dsq():
    vn = np.random.uniform(size=(1024, 1, 3))
    s = np.random.uniform(size=(41, 1, 1, 1))
    w = np.random.uniform(size=(1, 3))
    df = np.random.uniform(size=(1, 3))
    h = np.random.uniform(size=(1, 3))
    sPenalty = np.random.uniform(size=(41, 1, 1))

    old_result = old_parallel_dsq(vn, s, w, df, h, sPenalty)
    new_result = parallel_dsq(vn, s, w, df, h, sPenalty)
    assert np.allclose(old_result, new_result)
