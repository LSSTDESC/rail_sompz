class CellMap(object):
    # TODO: consider moving to another file so that sompz.py contains only
    # informers, estimators, and summarizers
    """This class will link to two Self-Organizing Maps.
       This class will contain functionality to infer n(z)
       given the two self organizing maps"""
    def __init__(self):
        # self.deep_som = None
        # self.wide_som = None
        # self.pcchat = None
        pass
    def read(cls, path, name=''):
        pass
    def write(cls, path, name=''):
        pass
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
