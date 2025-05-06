from rail.core.stage import RailStage, RailPipeline
from rail.utils.catalog_utils import CatalogConfigBase

from rail.estimation.algos.sompz import SOMPZInformer



class InformSomPZPipeline(RailPipeline):

    default_input_dict={
        'input_deep_data':'dummy.in',
        'input_wide_data':'dummy.in',
    }

    def __init__(self, wide_catalog_tag: str="SompzWideTestCatalogConfig", deep_catalog_tag: str="SompzDeepTestCatalogConfig", catalog_module: str="rail.sompz.utils"):
        RailPipeline.__init__(self)
        
        wide_catalog_class = CatalogConfigBase.get_class(wide_catalog_tag, catalog_module)
        deep_catalog_class = CatalogConfigBase.get_class(deep_catalog_tag, catalog_module)

        wide_config_dict = wide_catalog_class.build_base_dict()
        deep_config_dict = deep_catalog_class.build_base_dict()

        som_params_deep = dict(
            inputs=deep_config_dict['bands'],
            input_errs=deep_config_dict['err_bands'],
            zero_points=[30.]*len(deep_config_dict['bands']),
            convert_to_flux=True,
            set_threshold=True,
            thresh_val=1.e-5,
            som_shape=[32,32],
            som_minerror=0.005,
            som_take_log=False,
            som_wrap=False,
        )

        som_params_wide = dict(
            inputs=wide_config_dict['bands'],
            input_errs=wide_config_dict['err_bands'],
            zero_points=[30.]*len(wide_config_dict['bands']),                        
            convert_to_flux=True,
            som_shape=[25, 25],
            som_minerror=0.005,
            som_take_log=False,
            som_wrap=False,
        )
        
        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        # 1: train the deep SOM
        self.som_informer_deep = SOMPZInformer.build(
            aliases=dict(input_data='input_deep_data'),
            **som_params_deep,
        )

        # 2: train the wide SOM
        self.som_informer_wide = SOMPZInformer.build(
            aliases=dict(input_data='input_wide_data'),
            **som_params_wide,
        )
