
import os
import sys
import numpy as np
from rail.core.utils import RAILDIR
from rail.core import common_params
import tables_io
import matplotlib.pyplot as plt

import pandas as pd
import astropy.io.fits as fits

from rail.estimation.algos.sompz import SOMPZEstimatorWide, SOMPZEstimatorDeep
from rail.estimation.algos.sompz import SOMPZPzc, SOMPZPzchat, SOMPZPc_chat
from rail.estimation.algos.sompz import SOMPZTomobin, SOMPZnz

bands = ['u','g','r','i','z','y','J','H', 'F']
#bands = ['u','g','r','i','z','y']

deepbands = []
deeperrs = []
zeropts = []
widezeropts = []
for band in bands:
    deepbands.append(f'{band}')
    deeperrs.append(f'{band}_err')
    zeropts.append(30.)

widebands = []
wideerrs = []  
for band in bands[:6]:
    widebands.append(f'{band}')
    wideerrs.append(f'{band}_err')
    widezeropts.append(30.)


deep_som_params = dict(
    inputs=deepbands, 
    input_errs=deeperrs,
    hdf5_groupname="",
    zero_points=zeropts,
    som_shape=[32,32], # now a list instead of a tuple!
    som_minerror=0.01,
    som_take_log=False,
    convert_to_flux=True,
    set_threshold=True,
    thresh_val=1.e-5,
    thresh_val_err=1.e-5,
)


wide_som_params = dict(
    inputs=widebands, 
    input_errs=wideerrs,
    hdf5_groupname="",
    zero_points=widezeropts,
    som_shape=[25,25], # now a list instead of a tuple!
    som_minerror=0.005,
    som_take_log=False,
    convert_to_flux=True,
    set_threshold=True,
    thresh_val=1.e-5,
    thresh_val_err=1.e-5,
)

bin_edges_deep=[0.0,0.5,1.0,2.0,3.0]
zbins_min_deep=0.0
zbins_max_deep=3.2
zbins_dz_deep=0.02

bin_edges_tomo=[0.2, 0.6, 1.2, 1.8, 2.5]
zbins_min_tomo=0.0
zbins_max_tomo=3.0
zbins_dz_tomo=0.025

            
class EstimateSomPZPipeline(RailPipeline):

    default_input_dict={
        'wide_model': 'dummy.in',
        'deep_model': 'dummy.in',
        'input_spec_data':'dummy.in',
        'input_deep_data':'dummy.in',
        'wide_data':'dummy.in',
    }

    def __init__(self):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        active_catalog = CatalogConfigBase.active_class()

        # 1. Find the best cell mapping for all of the deep/balrog galaxies into the deep SOM
        self.som_deepdeep_estimator = SOMPZEstimatorDeep.build(
            aliases=dict(data="input_deep_data"),
            **deep_som_params,
        )
        
        # 2. Find the best cell mapping for all of the deep/balrog galaxies into the wide SOM
        self.som_deepwide_estimator = SOMPZEstimatorWide.build(
            aliases=dict(data="input_deep_data"),
            **wide_som_params,
        )            

        # 3. Find the best cell mapping for all of the spectrscopic galaxies into the deep SOM
        self.som_deepspec_estimator = SOMPZEstimatorDeep.build(
            aliases=dict(data="input_spec_data"),
            **deep_som_params
        )
        
        # 4. Use these cell assignments to compute the pz_c redshift histograms in deep SOM.
        # These distributions are redshift pdfs for individual deep SOM cells. 
        self.som_pzc = SOMPZPzc.build(
            redshift_col="redshift",
            bin_edges=bin_edges_deep,
            zbins_min=zbins_min_deep,
            zbins_max=zbins_max_deep,
            zbins_dz=zbins_dz_deep,
            deep_groupname="",
            aliases=dict(spec_data="input_spec_data"),
            connections=dict(cell_deep_spec_data=self.som_deepspec_estimator.io.assignment),
        )
        
        # 5. Compute the 'transfer function'.
        # The 'transfer function' weights relating deep to wide photometry.
        # These weights set the relative importance of p(z) from deep SOM cells for each
        # corresponding wide SOM cell.
        # These are traditionally made by injecting galaxies into images with Balrog.
        self.som_pcchat = SOMPZPc_chat.build(
            connections=dict(
                cell_deep_balrog_data=self.som_deepdeep_estimator.io.assignment,
                cell_wide_balrog_data=self.som.som_deepwide_estimator.io.assignment,
            )
        )
        
        # 6. Find the best cell mapping for all of the wide-field galaxies into the wide SOM
        self.som_widewide_estimator = SOMPZEstimatorWide.build(
            aliases=dict(data="input_wide_data"),
            **wide_som_params,
        )

        # 7. Compute more weights.
        # These weights represent the normalized occupation fraction of each wide SOM cell
        # relative to the full sample.
        self.som_pzchat = SOMPZPzchat.build(
            bin_edges=bin_edges_tomo,
            zbins_min=zbins_min_tomo,
            zbins_max=zbins_max_tomo,
            zbins_dz=zbins_dz_tomo,
            aliases=dict(
                spec_data='input_spec_data',
            )
            connections=dict(
                cell_deep_spec_data=self.som_deepspec_estimator.io.assignment,
                cell_wide_wide_data=self.som_widewide_estimator.io.assignment,
                pz_c=self.som_pzc.io.pz_c,
                pc_chat=self.som_pcchat.io.pc_chat,
            ),
        )

        # 8. Find the best cell mapping for all of the spectroscopic galaxies into the wide SOM
        self.som_widespec_estimator = SOMPZEstimatorWide.build(
            aliases=dict(data="input_spec_data"),
            **wide_som_params,
        )

        # 9. Define a tomographic bin mapping
        self.som_tomobin = SOMPZTomobin.build(
            bin_edges=bin_edges_tomo,
            zbins_min=zbins_min_tomo,
            zbins_max=zbins_max_tomo,
            zbins_dz=zbins_dz_tomo,
            wide_som_size=625,
            deep_som_size=1024,
            aliases=dict(
                spec_data='input_spec_data',
            ),
            connections=dict(
                cell_deep_spec_data=self.som_deepspec_estimator.io.assignment,
                cell_wide_spec_data=self.som_widespec_estimator.io.assignment,
            ),
        )

        # 10. Assemble the final tomographic bin estimates
        self.som_nz = SOMPZnz.build(
            bin_edges=bin_edges_tomo,
            zbins_min=zbins_min_tomo,
            zbins_max=zbins_max_tomo,
            zbins_dz=zbins_dz_tomo,
            redshift_col="redshift",
            aliases=dict(
                spec_data='input_spec_data',
            ),
            connections=dict(
                cell_deep_spec_data=self.som_deepspec_estimator.io.assignment,
                cell_wide_wide_data=self.som_widewide_estimator.io.assignment,
                tomo_bins_wide=self.som_tomobin.io.tomo_bin,
                pc_chat=self.som_pcchat.io.pc_chat,
            ),
        )
