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
        # TODO: initialize SOM of specified dimensions
        """
        self.TODO = TODO
        """

    def run(self):
        """train SOM and compute redshift 
        """
        # TODO: train Deep SOM with deep field (DF) data
        # TODO: train Wide SOM with wide field (WF) data
        # TODO: compute p(z|c), redshift distributions in deep SOM cells
        # TODO: compute p(c|chat), transfer function
        # relating deep SOM cells c to wide SOM cells chat

class SOMPZEstimator(CatEstimator):
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "SOMPZEstimator"
    config_options = CatEstimator.config_options.copy()
    """
    config_options.update(TODO)
    """
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

    def _initialize_run(self):
        super()._initialize_run()

        # If we are not the root process then we wait for
        # the root to (potentially) create all the templates before
        # reading them ourselves.
        if self.rank > 0:  # pragma: no cover
            # The Barrier method causes all processes to stop
            # until all the others have also reached the barrier.
            # If our rank is > 0 then we must be running under MPI.
            self.comm.Barrier()
            self.flux_templates = self._load_templates()
        # But if we are the root process then we just go
        # ahead and load them before getting to the Barrier,
        # which will allow the other processes to continue
        else:
            self.flux_templates = self._load_templates()
            # We might only be running in serial, so check.
            # If we are running MPI, then now we have created
            # the templates we let all the other processes that
            # stopped at the Barrier above continue and read them.
            if self.is_mpi():  # pragma: no cover
                self.comm.Barrier()

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        self.modeldict = self.model

    def _preprocess_magnitudes(self, data):
        bands = self.config.bands
        errs = self.config.err_bands

        fluxdict = {}
        
        # Load the magnitudes
        zp_frac = e_mag2frac(np.array(self.config.zp_errors))

        # replace non-detects with TODO
        for bandname, errname in zip(bands, errs):
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                detmask = np.isnan(data[bandname])
            else:
                detmask = np.isclose(data[bandname], self.config.nondetect_val)
            if isinstance(data, pd.DataFrame):
                data.loc[detmask, bandname] = 99.0
            else:
                data[bandname][detmask] = 99.0

        # replace non-observations with -99
        for bandname, errname in zip(bands, errs):
            if np.isnan(self.config.unobserved_val):  # pragma: no cover
                obsmask = np.isnan(data[bandname])
            else:
                obsmask = np.isclose(data[bandname], self.config.unobserved_val)
            if isinstance(data, pd.DataFrame):
                data.loc[obsmask, bandname] = -99.0
            else:
                data[bandname][obsmask] = -99.0


        # Only one set of mag errors
        mag_errs = np.array([data[er] for er in errs]).T

        # Group the magnitudes and errors into one big array
        mags = np.array([data[b] for b in bands]).T

        # Convert to pseudo-fluxes
        flux = 10.0**(-0.4 * mags)
        flux_err = flux * (10.0**(0.4 * mag_errs) - 1.0)

        # Convert to Lupton et al. 1999 magnitudes ('Luptitudes')
        # TODO
        
        # Upate the flux dictionary with new things we have calculated
        fluxdict['flux'] = flux
        fluxdict['flux_err'] = flux_err
        
        return fluxdict

    def _estimate_pdf(self, flux, flux_err):
        return #TODO

    def _process_chunk(self, start, end, data, first):
        """
        Run SOMPZ on a chunk of data
        """
        #TODO
