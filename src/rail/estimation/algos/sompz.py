"""
Port of SOMPZ
"""

import os
import numpy as np
import sys
from ceci.config import StageParameter as Param
from rail.core.data import TableHandle, ModelHandle
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.core.utils import RAILDIR
import rail.estimation.algos.som as somfuncs
from rail.core.common_params import SHARED_PARAMS


def_bands = ["u", "g", "r", "i", "z", "y"]
default_input_names = []
default_err_names = []
default_zero_points = []
for band in def_bands:
    default_input_names.append(f"mag_{band}_lsst")
    default_err_names.append(f"mag_err_{band}_lsst")
    default_zero_points.append(30.)


def mag2flux(mag, zero_pt=30):
    # zeropoint: M = 30 <=> f = 1
    exponent = (mag - zero_pt) / (-2.5)
    val = 1 * 10 ** (exponent)
    return val


def magerr2fluxerr(magerr, flux):
    coef = np.log(10) / -2.5
    return np.abs(coef * magerr * flux)


class SOMPZInformer(CatInformer):
    """Inform stage for SOMPZEstimator
    """
    name = "SOMPZInformer"
    config_options = CatInformer.config_options.copy()
    config_options.update(redshift_col=SHARED_PARAMS,
                          deep_groupname=Param(str, "photometry", msg="hdf5_groupname for deep data"),
                          wide_groupname=Param(str, "photometry", msg="hdf5_groupname for wide data"),
                          inputs_deep=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          input_errs_deep=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for deep data"),
                          inputs_wide=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for wide data"),
                          input_errs_wide=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for wide data"),
                          zero_points_deep=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for deep data, if needed"),
                          zero_points_wide=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for wide data, if needed"),
                          som_shape_deep=Param(tuple, (32, 32), msg="shape for the deep som, must be a 2-element tuple"),
                          som_shape_wide=Param(tuple, (32, 32), msg="shape for the wide som, must be a 2-element tuple"),
                          som_minerror_deep=Param(float, 0.01, msg="floor placed on observational error on each feature in deep som"),
                          som_minerror_wide=Param(float, 0.01, msg="floor placed on observational error on each feature in wide som"),
                          som_wrap_deep=Param(bool, False, msg="flag to set whether the deep SOM has periodic boundary conditions"),
                          som_wrap_wide=Param(bool, False, msg="flag to set whether the wide SOM has periodic boundary conditions"),
                          som_take_log_deep=Param(bool, True, msg="flag to set whether to take log of inputs (i.e. for fluxes) for deep som"),
                          som_take_log_wide=Param(bool, True, msg="flag to set whether to take log of inputs (i.e. for fluxes) for wide som"),
                          convert_to_flux_deep=Param(bool, False, msg="flag for whether to convert input columns to fluxes for deep data"
                                                     "set to true if inputs are mags and to False if inputs are already fluxes"),
                          convert_to_flux_wide=Param(bool, False, msg="flag for whether to convert input columns to fluxes for wide data"),
                          set_threshold_deep=Param(bool, False, msg="flag for whether to replace values below a threshold with a set number"),
                          thresh_val_deep=Param(float, 1.e-5, msg="threshold value for set_threshold for deep data"),
                          set_threshold_wide=Param(bool, False, msg="flag for whether to replace values below a threshold with a set number"),
                          thresh_val_wide=Param(float, 1.e-5, msg="threshold value for set_threshold for wide data"))

    #inputs = [('input_spec_data', TableHandle),
    #          ('input_deep_data', TableHandle),
    #          ('input_wide_data', TableHandle),
    #          ]

    inputs = [('input_deep_data', TableHandle),
              ('input_wide_data', TableHandle),
              ]
    
    # outputs = [('model_som_deep', ModelHandle),
    #            ('model_som_wide', ModelHandle)]
    ### outputs = [('model', ModelHandle)]

    def __init__(self, args, comm=None):
        """Init function, init config stuff
        """
        CatInformer.__init__(self, args, comm=comm)
        self.deep_model = None
        self.wide_model = None

    def run(self):

        # note: hdf5_groupname is a SHARED_PARAM defined in the parent class!
        if self.config.deep_groupname:
            deep_data = self.get_data('input_deep_data')[self.config.deep_groupname]
        else:  # pragma: no cover
        # DEAL with hdf5_groupname stuff later, just assume it's in the top level for now!
            deep_data = self.get_data('input_deep_data')
        wide_data = self.get_data('input_wide_data')[self.config.wide_groupname]
        # spec_data = self.get_data('input_spec_data')

        num_inputs_deep = len(self.config.inputs_deep)
        num_inputs_wide = len(self.config.inputs_wide)
        ngal_deep = len(deep_data[self.config.inputs_deep[0]])
        ngal_wide = len(wide_data[self.config.inputs_wide[0]])
        print(f"{ngal_deep} galaxies in deep sample")
        print(f"{ngal_wide} galaxies in wide sample")

        deep_input = np.zeros([ngal_deep, num_inputs_deep])
        deep_errs = np.zeros([ngal_deep, num_inputs_deep])
        wide_input = np.zeros([ngal_wide, num_inputs_wide])
        wide_errs = np.zeros([ngal_wide, num_inputs_wide])

        # assemble deep data
        for i, (col, errcol) in enumerate(zip(self.config.inputs_deep, self.config.input_errs_deep)):
            if self.config.convert_to_flux_deep:
                deep_input[:, i] = mag2flux(deep_data[col], self.config.zero_points_deep[i])
                deep_errs[:, i] = magerr2fluxerr(deep_data[errcol], deep_input[:, i])
            else:
                deep_input[:, i] = deep_data[col]
                deep_errs[:, i] = deep_data[errcol]

        # assemble wide data
        for i, (col, errcol) in enumerate(zip(self.config.inputs_wide, self.config.input_errs_wide)):
            if self.config.convert_to_flux_wide:
                wide_input[:, i] = mag2flux(wide_data[col], self.config.zero_points_deep[i])
                wide_errs[:, i] = magerr2fluxerr(wide_data[errcol], wide_input[:, i])
            else:
                wide_input[:, i] = wide_data[col]
                wide_errs[:, i] = wide_data[errcol]


        # put a temporary threshold bit in fix this up later...
        if self.config.set_threshold_deep:
            truncation_value = 1e-2
            for i in range(num_inputs_deep):
                mask = (deep_input[:, i] < self.config.thresh_val_deep)
                deep_input[:, i][mask] = truncation_value
                errmask = (deep_errs[:, i] < self.config.thresh_val_deep)
                deep_errs[:, i][errmask] = truncation_value

        if self.config.set_threshold_wide:
            truncation_value = 1e-2
            for i in range(num_inputs_deep):
                mask = (wide_input[:, i] < self.config.thresh_val_wide)
                wide_input[:, i][mask] = truncation_value
                errmask = (wide_errs[:, i] < self.config.thresh_val_wide)
                wide_errs[:, i][errmask] = truncation_value

        sommetric = somfuncs.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
        learn_func = somfuncs.hFunc(ngal_deep, sigma=(30, 1))

        print(f"Training deep SOM of shape {self.config.som_shape_deep}...")
        deep_som = somfuncs.NoiseSOM(sommetric, deep_input, deep_errs, learn_func,
                                     shape=self.config.som_shape_deep, minError=self.config.som_minerror_deep,
                                     wrap=self.config.som_wrap_deep, logF=self.config.som_take_log_deep)
        print(f"Training wide SOM of shape {self.config.som_shape_wide}...")
        learn_func = somfuncs.hFunc(ngal_wide, sigma=(30,1))
        wide_som = somfuncs.NoiseSOM(sommetric, wide_input, wide_errs, learn_func,
                                     shape=self.config.som_shape_wide, minError=self.config.som_minerror_wide,
                                     wrap=self.config.som_wrap_wide, logF=self.config.som_take_log_wide)

        model = dict(deep_som=deep_som, wide_som=wide_som, deep_columns=self.config.inputs_deep,
                     deep_err_columns=self.config.input_errs_deep, wide_columns=self.config.inputs_wide,
                     wide_err_columns=self.config.input_errs_wide)
        self.model = model

        """train SOMs
        """
        # TODO handle cfgfile io
        # TODO define arguments
        # TODO reconcile arguments given and expected
        # TODO connect kwargs with config file
        #with open(cfgfile, 'r') as fp:
        #    cfg = yaml.safe_load(fp)

        ## Read variables from config file
        #output_path = cfg['out_dir']
        #som_dim = cfg['deep_som_dim']
        #input_deep_balrog_file = cfg['deep_balrog_file']
        #bands = cfg['deep_bands']
        #bands_label = cfg['deep_bands_label']
        #bands_err_label = cfg['deep_bands_err_label']
        
        #deep_balrog_data = fitsio.read(input_deep_balrog_file) # reconcile with TableHandle

        ## Create flux and flux_err vectors
        #len_deep = len(deep_balrog_data[bands_label + bands[0]])
        #fluxes_d = np.zeros((len_deep, len(bands)))
        #fluxerrs_d = np.zeros((len_deep, len(bands)))

        #for i, band in enumerate(bands):
        #    print(i, band)
        #    fluxes_d[:, i] = deep_balrog_data[bands_label + band]
        #    fluxerrs_d[:, i] = deep_balrog_data[bands_err_label + band]

        ## Train the SOM with this set (takes a few hours on laptop!)
        #nTrain = fluxes_d.shape[0]

        ## Scramble the order of the catalog for purposes of training
        #indices = np.random.choice(fluxes_d.shape[0], size=nTrain, replace=False)

        ## Some specifics of the SOM training
        #hh = ns.hFunc(nTrain, sigma=(30, 1))
        #metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)

        ## Now training the SOM 
        #deep_som = ns.NoiseSOM(metric, fluxes_d[indices, :], fluxerrs_d[indices, :],
        #                       learning=hh,
        #                       shape=(som_dim, som_dim),
        #                       wrap=False, logF=True,
        #                       initialize='sample',
        #                       minError=0.02)
        
        #self.add_data('model_som_deep',  deep_som)
        #self.add_data('model_som_wide',  deep_som)
        self.add_data('model', self.model)

        # TAKE OUT SPECZ DATA FOR NOW, I DON'T KNOW IF IT'S ACTUALLY USED HERE!
        #    def inform(self, input_spec_data, input_deep_data, input_wide_data):
    def inform(self, input_deep_data, input_wide_data):
        #self.add_data('input_spec_data', input_spec_data)
        self.set_data('input_deep_data', input_deep_data)
        self.set_data('input_wide_data', input_wide_data)
        
        self.run()
        self.finalize()

        #return dict(model_som_deep=self.get_handle('model_som_deep'),
        #            model_som_wide=self.get_handle('model_som_wide'))
        return self.model
        
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
