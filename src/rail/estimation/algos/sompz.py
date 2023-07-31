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
from rail.core.common_params import SHARED_PARAMS

def nzfunc(z, z0, alpha, km, m, m0):  # pragma: no cover
    zm = z0 + (km * (m - m0))
    return np.power(z, alpha) * np.exp(-1. * np.power((z / zm), alpha))


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
        """
        self.TODO = TODO
        """

    def run(self):
        """compute the best fit prior parameters
        """


class SOMPZEstimator(CatEstimator):
    """CatEstimator subclass to implement basic marginalized PDF for SOMPZ
    In addition to the marginalized redshift PDF, we also compute several
    ancillary quantities that will be stored in the ensemble ancil data:
    zmode: mode of the PDF
    amean: mean of the PDF
    """
    name = "SOMPZEstimator"
    config_options = CatEstimator.config_options.copy()
    """
    config_options.update(TODO)
    """
    def __init__(self, args, comm=None):
        """Constructor, build the CatEstimator, then do BPZ specific setup
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

    def _load_templates(self):
        from desc_sompz.useful_py3 import get_str, get_data, match_resol

        # The redshift range we will evaluate on
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        z = self.zgrid

        data_path = self.data_path
        columns_file = self.config.columns_file
        ignore_rows = ["M_0", "OTHER", "ID", "Z_S"]
        filters = [f for f in get_str(columns_file, 0) if f not in ignore_rows]

        spectra_file = os.path.join(data_path, "SED", self.config.spectra_file)
        spectra = [s[:-4] for s in get_str(spectra_file)]

        nt = len(spectra)
        nf = len(filters)
        nz = len(z)
        flux_templates = np.zeros((nz, nt, nf))

        ab_dir = os.path.join(data_path, "AB")
        os.makedirs(ab_dir, exist_ok=True)

        TODO 
        return flux_templates

    def _preprocess_magnitudes(self, data):
        from desc_sompz.sompz_tools_py3 import e_mag2frac

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

        # Check if an object is seen in each band at all.
        # Fluxes not seen at all are listed as infinity in the input,
        # so will come out as zero flux and zero flux_err.
        # Check which is which here, to use with the ZP errors below
        seen1 = (flux > 0) & (flux_err > 0)
        seen = np.where(seen1)
        # unseen = np.where(~seen1)
        # replace Joe's definition with more standard SOMPZ style
        nondetect = 99.
        nondetflux = 10.**(-0.4 * nondetect)
        unseen = np.isclose(flux, nondetflux, atol=nondetflux * 0.5)

        # replace mag = 99 values with 0 flux and 1 sigma limiting magnitude
        # value, which is stored in the mag_errs column for non-detects
        # NOTE: We should check that this same convention will be used in
        # LSST, or change how we handle non-detects here!
        flux[unseen] = 0.
        flux_err[unseen] = 10.**(-0.4 * np.abs(mag_errs[unseen]))

        # Add zero point magnitude errors.
        # In the case that the object is detected, this
        # correction depends onthe flux.  If it is not detected
        # then SOMPZ uses half the errors instead
        add_err = np.zeros_like(flux_err)
        add_err[seen] = ((zp_frac * flux)**2)[seen]
        add_err[unseen] = ((zp_frac * 0.5 * flux_err)**2)[unseen]
        flux_err = np.sqrt(flux_err**2 + add_err)

        # Convert non-observed objects to have zero flux
        # and enormous error, so that their likelihood will be
        # flat. This follows what's done in the sompz script.
        nonobserved = -99.
        unobserved = np.isclose(mags, nonobserved)
        flux[unobserved] = 0.0
        flux_err[unobserved] = 1e108

        # Upate the flux dictionary with new things we have calculated
        fluxdict['flux'] = flux
        fluxdict['flux_err'] = flux_err
        m_0_col = self.config.bands.index(self.config.ref_band)
        fluxdict['mag0'] = mags[:, m_0_col]
        
        return fluxdict

    def _estimate_pdf(self, flux_templates, kernel, flux, flux_err, mag_0, z):
        from desc_sompz.sompz_tools_py3 import p_c_z_t
        #from desc_sompz.prior_from_dict import prior_function
        return TODO

    def _process_chunk(self, start, end, data, first):
        """
        Run SOMPZ on a chunk of data
        """
        TODO
