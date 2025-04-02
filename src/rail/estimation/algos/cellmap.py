import numpy as np
import pandas as pd
import h5py


class CellMap(object):
    """This class will link to two Self-Organizing Maps.
       This class will contain functionality to infer n(z)
       given the two self organizing maps"""

    def __init__(self, data, wide_som, deep_som, pcchat, **kwargs):
        """Initialize the CellMap Object

        Parameters
        ----------
        data
            A pandas dataframe of galaxies with both deep and
            wide observations. It must contain the
            wide_columns, wide_err_columns, deep_columns, and
            deep_err_columns.

        wide_som
            SelfOrganizingMap object trained on the wide fluxes

        deep_som
            SelfOrganizingMap object trained on the deep fluxes

        pcchat
            An array of shape (n_tomo_bins, `prod(*som.map_shape)`,
            `prod(*som.map_shape)`) which gives the probability
            of a galaxy really being in cell c given that it is
            in cell chat. This may or may not have come from
            data -- we could train this on some other sample.
        """
        # TODO determine best place to implement overlap_weight
        # TODO determine best place to implement column names
        # TODO determine best place to implement mag zero point dependence
        self.data = data

        self.wide_som = wide_som
        self.deep_som = deep_som
        self.pcchat = pcchat

        self.wide_columns = wide_columns # TODO this and following lines won't work
        self.wide_err_columns = wide_err_columns
        self.deep_columns = deep_columns
        self.deep_err_columns = deep_err_columns

        self.kwargs = {}
        self.kwargs.update(kwargs)

    @classmethod
    def read(cls, path, name='cellmap'):
        import sompz
        import pandas as pd

        print('reading pcchat...')
        # pcchat
        try:
            with h5py.File(path, 'r') as h5f:
                pcchat = h5f['{0}/pcchat'.format(name)][:]
                print('...success')
        except:
            assert False,'TODO'
        data = pd.read_hdf(path, '{0}/data'.format(name))

        overlap_weight = None

        print('reading columns...')
        try:
            # try reading with h5py instead
            with h5py.File(path, 'r') as h5f:
                deep_columns = h5f['{0}/deep_columns'.format(name)][:].tolist()
                deep_columns = [c.decode('utf-8') for c in deep_columns]
                deep_err_columns = h5f['{0}/deep_err_columns'.format(name)][:].tolist()
                deep_err_columns = [c.decode('utf-8') for c in deep_err_columns]
                wide_columns = h5f['{0}/wide_columns'.format(name)][:].tolist()
                wide_columns = [c.decode('utf-8') for c in wide_columns]
                wide_err_columns = h5f['{0}/wide_err_columns'.format(name)][:].tolist()
                wide_err_columns = [c.decode('utf-8') for c in wide_err_columns]
                kind = h5f['{0}/type'.format(name)][:].tolist()[0]
                kind = kind.decode('utf-8')
                print('...success')
        except:
            assert False, 'TODO'

        print('reading SOMs...')
        try:
            wide_som = SelfOrganizingMap.read(path, name='{0}/wide_som'.format(name))
            deep_som = SelfOrganizingMap.read(path, name='{0}/deep_som'.format(name))
            print('...success')
        except KeyError:
            assert False, 'TODO'
        cm_class = getattr(sompz, kind)
        cm = cm_class(data, wide_som, deep_som, pcchat, **kwargs)
        return cm

    def write(self, path, name='cellmap'):
        self.data.to_hdf(path, '{0}/data'.format(name))

        for key in self.kwargs:
            pd.Series([self.kwargs[key]]).to_hdf(path, '{0}/kwargs/{1}'.format(name, key))

        self.wide_som.write(path, name='{0}/wide_som'.format(name))
        self.deep_som.write(path, name='{0}/deep_som'.format(name))

        with h5py.File(path, 'r+') as h5f:
            if self.pcchat is not None:
                try:
                    h5f.create_dataset('{0}/pcchat'.format(name), data=self.pcchat)
                except RuntimeError:
                    # path already exists
                    h5f['{0}/pcchat'.format(name)][...] = self.pcchat

            for ci in [['wide_columns', self.wide_columns], ['deep_columns', self.deep_columns],
                       ['wide_err_columns', self.wide_err_columns], ['deep_err_columns', self.deep_err_columns],
                       ['type', [self.__class__.__name__]]]:
                label, data_raw = ci
                data = [x.encode('utf-8') for x in data_raw]
                col = '{0}/{1}'.format(name, label)
                try:
                    h5f.create_dataset(col, data=data)
                except RuntimeError:
                    # path already exists. But we don't want to save because we have no idea if the column is long enough.
                    del h5f[col]
                    h5f.create_dataset(col, data=data)
    def update(self, data=None,
               overlap_weight=None, pcchat=None,
               wide_som=None, deep_som=None,
               wide_columns=None, deep_columns=None, wide_err_columns=None, deep_err_columns=None):
        """Returns a new cellmap with new data and pcchat. Everything is copied
        """
        # checking all the variables. There probably is a better way
        if data is None:
            data = self.data
        if pcchat is None:
            if self.pcchat is None:
                # None can't copy itself
                pcchat = None
            else:
                pcchat = self.pcchat
        if wide_som is None:
            wide_som = self.wide_som
        if deep_som is None:
            deep_som = self.deep_som
        if wide_columns is None:
            wide_columns = self.wide_columns
        if wide_err_columns is None:
            wide_err_columns = self.wide_err_columns
        if deep_columns is None:
            deep_columns = self.deep_columns
        if deep_err_columns is None:
            deep_err_columns = self.deep_err_columns

        # in read, we can just do cls(kwargs), but the self is the actual object, so we have to do self.__class__. Note that you can NOT do cls.__class__
        new_cm = self.__class__(data.copy(), overlap_weight, wide_som.copy(), deep_som.copy(), pcchat.copy(), wide_columns, wide_err_columns, deep_columns, deep_err_columns)
        return new_cm
        
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
    def calculate_pcchat(self, balrog_data, max_iter=0, force_assignment=True, replace=False):
        """With a given balrog_data, calculate a new pcchat.
           By default, returns a new cmap object with the new pcchat.
           If replace=True, replaces the pcchat object in self."""

        print('Starting construction of new p(c|chat,s) from new s. Loading data.')

        if force_assignment or deep_cell_key not in balrog_data.columns:
            print('Assigning SOM Deep')
            cell_deep = self.assign_deep(balrog_data)
        else:
            cell_deep = balrog_data[deep_cell_key].values

        if force_assignment or wide_cell_key not in balrog_data.columns:
            print('Assigning SOM wide')
            cell_wide = self.assign_wide(balrog_data)
        else:
            cell_wide = balrog_data[wide_cell_key].values

        pcchat = self.build_pcchat(cell_wide, cell_deep,
                                   wide_som=self.wide_som,
                                   deep_som=self.deep_som,
                                   max_iter=max_iter,
                                   replace=replace)

        print('Creating class object')
        new_cm = self.update(data=self.data, pcchat=pcchat)
        return new_cm
class CellMapDESY3(CellMap):
    @classmethod
    def fit(cls, spec_data,
            overlap_weight, wide_columns, wide_err_columns,
            deep_columns, deep_err_columns,
            data_train_deep, data_train_wide,
            zp, deep_kwargs={}, wide_kwargs={}, **kwargs):
        # overlap_weight: Weights of galaxies in spec_data, to account for shear response, or uneven number of times these galaxies were drawn in the wide data

        t0 = time.time()

        deep_kwargs.update(kwargs)
        wide_kwargs.update(kwargs)

        deep_diag_ivar = True
        cls.zp = zp
        log('Fitting SOM to deep data', t0)
        deep_x = cls.get_x_deep(data_train_deep, deep_columns, zp)
        deep_ivar = cls.get_ivar_deep(data_train_deep, deep_columns, deep_err_columns)
        deep_som = SelfOrganizingMap.fit(deep_x, deep_ivar, diag_ivar=deep_diag_ivar, **deep_kwargs)

        log('Fitting SOM to wide data', t0)
        wide_x = cls.get_x_wide(data_train_wide, wide_columns, zp)
        wide_ivar = cls.get_ivar_wide(data_train_wide, wide_columns, wide_err_columns, zp)
        wide_som = SelfOrganizingMap.fit(wide_x, wide_ivar, **wide_kwargs)

        # get deep columns
        log('Loading spec_data', t0)
        deep_x = cls.get_x_deep(spec_data, deep_columns, zp)
        deep_ivar = cls.get_ivar_deep(spec_data, deep_columns, deep_err_columns)
        # do deep assignments
        log('Assigning SOM Deep', t0)
        cell_deep = deep_som.assign(deep_x, deep_ivar, diag_ivar=deep_diag_ivar)
        spec_data['cell_deep'] = cell_deep

        pcchat = 0

        log('Creating class object', t0)
        return cls(spec_data, overlap_weight, wide_som, deep_som, pcchat, wide_columns, wide_err_columns, deep_columns, deep_err_columns, zp, **kwargs)
