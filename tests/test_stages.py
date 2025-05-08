from rail.core.stage import RailStage
from rail.core.data import Hdf5Handle
from rail.utils.catalog_utils import CatalogConfigBase

from rail.estimation.algos.sompz import (
    SOMPZInformer,
    SOMPZEstimatorWide,
    SOMPZEstimatorDeep,
    SOMPZPzc,
    SOMPZPc_chat,
    SOMPZPzchat,
    SOMPZTomobin,
    SOMPZnz,
)


deep_catalog_tag: str = "SompzDeepTestCatalogConfig"
catalog_module: str = "rail.sompz.utils"
deep_catalog_class = CatalogConfigBase.get_class(deep_catalog_tag, catalog_module)
deep_config_dict = deep_catalog_class.build_base_dict()

som_params_deep = dict(
    inputs=deep_config_dict["bands"],
    input_errs=deep_config_dict["err_bands"],
    zero_points=[30.0] * len(deep_config_dict["bands"]),
    convert_to_flux=True,
    set_threshold=True,
    thresh_val=1.0e-5,
    som_shape=[32, 32],
    som_minerror=0.005,
    som_take_log=False,
    som_wrap=False,
)


wide_catalog_tag: str = "SompzWideTestCatalogConfig"
catalog_module: str = "rail.sompz.utils"
wide_catalog_class = CatalogConfigBase.get_class(wide_catalog_tag, catalog_module)
wide_config_dict = wide_catalog_class.build_base_dict()

som_params_wide = dict(
    inputs=wide_config_dict["bands"],
    input_errs=wide_config_dict["err_bands"],
    zero_points=[30.0] * len(wide_config_dict["bands"]),
    convert_to_flux=True,
    som_shape=[25, 25],
    som_minerror=0.005,
    som_take_log=False,
    som_wrap=False,
)

bin_edges_deep = [0.0, 0.5, 1.0, 2.0, 3.0]
zbins_min_deep = 0.0
zbins_max_deep = 3.2
zbins_dz_deep = 0.02

bin_edges_tomo = [0.2, 0.6, 1.2, 1.8, 2.5]
zbins_min_tomo = 0.0
zbins_max_tomo = 3.0
zbins_dz_tomo = 0.025


def test_informer_deep(get_data):
    assert get_data == 0

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()

    som_informer_deep = SOMPZInformer.make_stage(
        name="test_informer_deep",
        **som_params_deep,
    )

    input_data_deep = DS.read_file(
        "input_data_deep",
        handle_class=Hdf5Handle,
        path="tests/romandesc_deep_data_37c_noinf.hdf5",
    )
    results = som_informer_deep.inform(input_data_deep)


def test_informer_wide(get_data):
    assert get_data == 0

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()

    som_informer_wide = SOMPZInformer.make_stage(
        name="test_informer_wide",
        **som_params_wide,
    )

    input_data_wide = DS.read_file(
        "input_data_wide",
        handle_class=Hdf5Handle,
        path="tests/romandesc_wide_data_50c_noinf.hdf5",
    )
    results = som_informer_wide.inform(input_data_wide)


def test_deepdeep_estimator(get_data, get_intermediates):
    assert get_data == 0
    assert get_intermediates == 0

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()

    som_deepdeep_estimator = SOMPZEstimatorDeep.make_stage(
        name="test_deepdeep_estimator",
        model="tests/intermediates/model_som_informer_deep.pkl",
        hdf5_groupname="",
        **som_params_deep,
    )

    input_data_deep = DS.read_file(
        "input_data_deep",
        handle_class=Hdf5Handle,
        path="tests/romandesc_deep_data_37c_noinf.hdf5",
    )
    results = som_deepdeep_estimator.estimate(input_data_deep)


def test_deepwide_estimator(get_data, get_intermediates):

    assert get_data == 0
    assert get_intermediates == 0

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()

    som_deepwide_estimator = SOMPZEstimatorWide.make_stage(
        name="test_deepwide_estimator",
        model="tests/intermediates/model_som_informer_wide.pkl",
        hdf5_groupname="",
        **som_params_wide,
    )

    input_data_wide = DS.read_file(
        "input_data_wide",
        handle_class=Hdf5Handle,
        path="tests/romandesc_wide_data_50c_noinf.hdf5",
    )
    results = som_deepwide_estimator.estimate(input_data_wide)


def test_pz_c(get_data, get_intermediates):

    assert get_data == 0
    assert get_intermediates == 0

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()

    som_pzc = SOMPZPzc.make_stage(
        name="test_pzc",
        redshift_col="redshift",
        bin_edges=bin_edges_deep,
        zbins_min=zbins_min_deep,
        zbins_max=zbins_max_deep,
        zbins_dz=zbins_dz_deep,
        deep_groupname="",
    )

    input_data_spec = DS.read_file(
        "input_data_spec",
        handle_class=Hdf5Handle,
        path="tests/romandesc_spec_data_18c_noinf.hdf5",
    )
    cell_deep_spec_data = DS.read_file(
        "cell_deep_spec_data",
        handle_class=Hdf5Handle,
        path="tests/intermediates/assignment_som_deepspec_estimator.hdf5",
    )

    result = som_pzc.estimate(input_data_spec, cell_deep_spec_data)


def test_pc_chat(get_intermediates):

    assert get_intermediates == 0

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()

    som_pcchat = SOMPZPc_chat.make_stage(
        name="test_pcchat",
    )

    cell_deep_balrog_data = DS.read_file(
        "cell_deep_balrog_data",
        handle_class=Hdf5Handle,
        path="tests/intermediates/assignment_som_deepdeep_estimator.hdf5",
    )
    cell_wide_balrog_data = DS.read_file(
        "cell_wide_balrog_data",
        handle_class=Hdf5Handle,
        path="tests/intermediates/assignment_som_deepwide_estimator.hdf5",
    )

    result = som_pcchat.estimate(cell_deep_balrog_data, cell_wide_balrog_data)


def test_pz_chat(get_data, get_intermediates):

    assert get_data == 0
    assert get_intermediates == 0

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()

    som_pzchat = SOMPZPzchat.make_stage(
        name="test_pzchat",
        bin_edges=bin_edges_tomo,
        zbins_min=zbins_min_tomo,
        zbins_max=zbins_max_tomo,
        zbins_dz=zbins_dz_tomo,
        redshift_col="redshift",
    )

    input_data_spec = DS.read_file(
        "input_data_spec",
        handle_class=Hdf5Handle,
        path="tests/romandesc_spec_data_18c_noinf.hdf5",
    )
    cell_deep_spec_data = DS.read_file(
        "cell_deep_spec_data",
        handle_class=Hdf5Handle,
        path="tests/intermediates/assignment_som_deepspec_estimator.hdf5",
    )
    cell_wide_wide_data = DS.read_file(
        "cell_wide_wide_data",
        handle_class=Hdf5Handle,
        path="tests/intermediates/assignment_som_widewide_estimator.hdf5",
    )
    pz_c = DS.read_file(
        "pz_c", handle_class=Hdf5Handle, path="tests/intermediates/pz_c_som_pzc.hdf5"
    )
    pc_chat = DS.read_file(
        "pc_chat",
        handle_class=Hdf5Handle,
        path="tests/intermediates/pc_chat_som_pcchat.hdf5",
    )

    result = som_pzchat.estimate(
        input_data_spec, cell_deep_spec_data, cell_wide_wide_data, pz_c, pc_chat
    )


def test_tomo_bin(get_data, get_intermediates):

    assert get_data == 0
    assert get_intermediates == 0

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()

    som_tomobin = SOMPZTomobin.make_stage(
        name="test_tomobin",
        bin_edges=bin_edges_tomo,
        zbins_min=zbins_min_tomo,
        zbins_max=zbins_max_tomo,
        zbins_dz=zbins_dz_tomo,
        wide_som_size=625,
        deep_som_size=1024,
        redshift_col="redshift",
    )

    input_data_spec = DS.read_file(
        "input_data_spec",
        handle_class=Hdf5Handle,
        path="tests/romandesc_spec_data_18c_noinf.hdf5",
    )
    cell_deep_spec_data = DS.read_file(
        "cell_deep_spec_data",
        handle_class=Hdf5Handle,
        path="tests/intermediates/assignment_som_deepspec_estimator.hdf5",
    )
    cell_wide_spec_data = DS.read_file(
        "cell_wide_spec_data",
        handle_class=Hdf5Handle,
        path="tests/intermediates/assignment_som_widespec_estimator.hdf5",
    )

    result = som_tomobin.estimate(
        input_data_spec, cell_deep_spec_data, cell_wide_spec_data
    )


def test_nz(get_data, get_intermediates):

    assert get_data == 0
    assert get_intermediates == 0

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()

    som_nz = SOMPZnz.make_stage(
        name="test_nz",
        bin_edges=bin_edges_tomo,
        zbins_min=zbins_min_tomo,
        zbins_max=zbins_max_tomo,
        zbins_dz=zbins_dz_tomo,
        redshift_col="redshift",
    )

    input_data_spec = DS.read_file(
        "input_data_spec",
        handle_class=Hdf5Handle,
        path="tests/romandesc_spec_data_18c_noinf.hdf5",
    )
    cell_deep_spec_data = DS.read_file(
        "cell_deep_spec_data",
        handle_class=Hdf5Handle,
        path="tests/intermediates/assignment_som_deepspec_estimator.hdf5",
    )
    cell_wide_wide_data = DS.read_file(
        "cell_wide_wide_data",
        handle_class=Hdf5Handle,
        path="tests/intermediates/assignment_som_widewide_estimator.hdf5",
    )
    tomo_bins_wide = DS.read_file(
        "tomo_bins_wide",
        handle_class=Hdf5Handle,
        path="tests/intermediates/tomo_bins_wide_som_tomobin.hdf5",
    )
    pc_chat = DS.read_file(
        "pc_chat",
        handle_class=Hdf5Handle,
        path="tests/intermediates/pc_chat_som_pcchat.hdf5",
    )

    result = som_nz.estimate(
        input_data_spec,
        cell_deep_spec_data,
        cell_wide_wide_data,
        tomo_bins_wide,
        pc_chat,
    )
