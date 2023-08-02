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
    def __init__(self, args, comm=None):
        """Init function, init config stuff
        """
        CatInformer.__init__(self, args, comm=comm)
        # TODO: initialize CellMap containing
        # SOMs of specified dimensions
        """
        self.TODO = TODO
        """

    def run(self):
        """train SOM and compute redshift 
        """
        # TODO: train Deep SOM with deep field (DF) data
        # this is a method run on a som instance
        # TODO: train Wide SOM with wide field (WF) data
        # this is a method run on a som instance
        
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
