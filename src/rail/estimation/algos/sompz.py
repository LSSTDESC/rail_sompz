"""
Port of SOMPZ
"""

import os
import numpy as np
import scipy.optimize as sciop
import pandas as pd
import scipy.integrate
import glob
import qp

#import pickle
import sys
import yaml
from sompz import NoiseSOM as ns

import fitsio

import tables_io
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.core.utils import RAILDIR
from rail.sompz.utils import RAIL_SOMPZ_DIR
from rail.sompz.utils import * # TODO
from rail.core.common_params import SHARED_PARAMS

class SOMPZInformer(CatInformer):
    """Inform stage for SOMPZEstimator, 
    """
    name = "SOMPZInformer"
    config_options = CatInformer.config_options.copy()
    """
    config_options.update(TODO)
    """
    inputs = [('input_spec_data', TableHandle),
              ('input_deep_data', TableHandle),
              ('input_wide_data', TableHandle),
              ]
    outputs = [('model_som_deep', ModelHandle),
               ('model_som_wide', ModelHandle)]
    
    def __init__(self, args, comm=None):
        """Init function, init config stuff
        """
        CatInformer.__init__(self, args, comm=comm)
        """
        self.TODO = TODO
        """
    def run(self):
        """train SOMs
        """
        # TODO handle cfgfile io
        # TODO define arguments
        # TODO reconcile arguments given and expected
        # TODO connect kwargs with config file
        with open(cfgfile, 'r') as fp:
            cfg = yaml.safe_load(fp)

        # Read variables from config file
        output_path = cfg['out_dir']
        som_dim = cfg['deep_som_dim']
        input_deep_balrog_file = cfg['deep_balrog_file']
        bands = cfg['deep_bands']
        bands_label = cfg['deep_bands_label']
        bands_err_label = cfg['deep_bands_err_label']
        
        deep_balrog_data = fitsio.read(input_deep_balrog_file) # reconcile with TableHandle

        # Create flux and flux_err vectors
        len_deep = len(deep_balrog_data[bands_label + bands[0]])
        fluxes_d = np.zeros((len_deep, len(bands)))
        fluxerrs_d = np.zeros((len_deep, len(bands)))

        for i, band in enumerate(bands):
            print(i, band)
            fluxes_d[:, i] = deep_balrog_data[bands_label + band]
            fluxerrs_d[:, i] = deep_balrog_data[bands_err_label + band]

        # Train the SOM with this set (takes a few hours on laptop!)
        nTrain = fluxes_d.shape[0]

        # Scramble the order of the catalog for purposes of training
        indices = np.random.choice(fluxes_d.shape[0], size=nTrain, replace=False)

        # Some specifics of the SOM training
        hh = ns.hFunc(nTrain, sigma=(30, 1))
        metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)

        # Now training the SOM 
        deep_som = ns.NoiseSOM(metric, fluxes_d[indices, :], fluxerrs_d[indices, :],
                               learning=hh,
                               shape=(som_dim, som_dim),
                               wrap=False, logF=True,
                               initialize='sample',
                               minError=0.02)
        
        self.add_data('model_som_deep',  deep_som)
        self.add_data('model_som_wide',  wide_som)

    def inform(self, input_spec_data, input_deep_data, input_wide_data):
        self.add_data('input_spec_data', input_spec_data)
        self.add_data('input_deep_data', input_deep_data)
        self.add_data('input_wide_data', input_wide_data)
        
        self.run()
        self.finalize()

        return dict(model_som_deep=self.get_handle('model_som_deep'),
                    model_som_wide=self.get_handle('model_som_wide'))
        
class SOMPZEstimator(CatEstimator):
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "SOMPZEstimator"
    config_options = CatEstimator.config_options.copy()
    def __init__(self, args, comm=None):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        CatEstimator.__init__(self, args, comm=comm)

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

        # check on bands, errs, and prior band
        if len(self.config.bands) != len(self.config.err_bands):  # pragma: no cover
            raise ValueError("Number of bands specified in bands must be equal to number of mag errors specified in err_bands!")
        if self.config.ref_band not in self.config.bands:  # pragma: no cover
            raise ValueError(f"reference band not found in bands specified in bands: {str(self.config.bands)}")

    def _estimate_pdf(self, flux, flux_err):
        # TODO: compute p(z|c), redshift distributions in deep SOM cells

        # TODO: compute p(c|chat), transfer function
        # relating deep SOM cells c to wide SOM cells chat
        cm = cm.calculate_pcchat(balrog_data, w, force_assignment=False, wide_cell_key='cell_wide_unsheared')

        # TODO: compute p(chat), occupation in wide SOM cells
        # TODO: compute p(z|chat) \propto sum_c p(z|c) p(c|chat)
        # TODO: construct tomographic bins bhat = {chat}
        # TODO: compute p(z|bhat) \propto sum_chat p(z|chat)


        return

    def _process_chunk(self, start, end, data, first):
        """
        Run SOMPZ on a chunk of data
        """
        #TODO
