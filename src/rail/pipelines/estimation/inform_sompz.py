#!/usr/bin/env python
# coding: utf-8

import ceci

# Various rail modules
from rail.core.stage import RailStage, RailPipeline
from rail.utils.catalog_utils import CatalogConfigBase


bands = ['u','g','r','i','z','y','J','H','F']

deepbands = []
deeperrs = []
zeropts = []
for band in bands:
    deepbands.append(f'{band}')
    deeperrs.append(f'{band}_err')
    zeropts.append(30.)

widebands = []
wideerrs = []  
for band in bands[:6]:
    widebands.append(f'{band}')
    wideerrs.append(f'{band}_err')
    
refband_deep=deepbands[3]
refband_wide=widebands[3]


som_params_deep = dict(
    inputs=deepbands,
    input_errs=deeperrs,
    zero_points=zeropts, 
    convert_to_flux=True,
    set_threshold=True, 
    thresh_val=1.e-5,
    som_shape=[32,32],
    som_minerror=0.005,
    som_take_log=False,
    som_wrap=False,
)

som_params_wide = dict(
    inputs=widebands,
    input_errs=wideerrs,
    convert_to_flux=True, 
    som_shape=[25, 25],
    som_minerror=0.005,
    som_take_log=False,
    som_wrap=False,
)


class InformSomPZPipeline(RailPipeline):

    default_input_dict={
        'input_deep_data':'dummy.in',
        'input_wide_data':'dummy.in',
    }

    def __init__(self):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        active_catalog = CatalogConfigBase.active_class()

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


        
