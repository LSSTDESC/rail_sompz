
from rail.core.stage import RailStage
from rail.core.data import Hdf5Handle
from rail.utils.catalog_utils import CatalogConfigBase

from rail.estimation.algos.sompz import SOMPZInformer


deep_catalog_tag: str = "SompzDeepTestCatalogConfig"
catalog_module: str = "rail.sompz.utils"
deep_catalog_class = CatalogConfigBase.get_class(
    deep_catalog_tag, catalog_module
)
deep_config_dict = deep_catalog_class.build_base_dict()

wide_catalog_tag: str = "SompzWideTestCatalogConfig"
catalog_module: str = "rail.sompz.utils"
wide_catalog_class = CatalogConfigBase.get_class(
    wide_catalog_tag, catalog_module
)
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



def test_informer_deep(get_data):
    assert get_data == 0

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()
    
    som_informer_deep = SOMPZInformer.make_stage(
        name='test_informer_deep',
        **som_params_deep,
    )

    input_data_deep = DS.read_file('input_data_deep', handle_class=Hdf5Handle, path='tests/romandesc_deep_data_37c_noinf.hdf5')    
    results = som_informer_deep.inform(input_data_deep)



def test_informer_wide(get_data):
    assert get_data == 0

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()
    
    som_informer_wide = SOMPZInformer.make_stage(
        name='test_informer_wide',
        **som_params_wide,
    )

    input_data_wide = DS.read_file('input_data_wide', handle_class=Hdf5Handle, path='tests/romandesc_wide_data_50c_noinf.hdf5')    
    results = som_informer_wide.inform(input_data_wide)

    
def xx_deepdeep_estimator(get_data, get_intermediates):
    assert get_data == 0
    assert get_intermediates == 0
    
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()
    
    som_deepdeep_estimator = SOMPZEstimatorDeep.make_stage(
        name='test_deepdeep_estimator',
        model='tests/',
        **som_params_deep,
    )

    input_data_deep = DS.read_file('input_data_deep', handle_class=Hdf5Handle, path='tests/romandesc_wide_data_50c_noinf.hdf5')    
    results = som_informer_wide.inform(input_data_wide)
