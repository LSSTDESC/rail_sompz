        import pandas as pd
        import h5py

class SelfOrganizingMap(object):
    def __init__(self, w, shape=None):
        """A class to encapsulate the SOM functions written out below, and to
        hold together the resultant variables

        Parameters
        ----------
        w :     self organizing map weights (*map_shapes, input_dims)
        shape : shape of the self organizing map (map_dim_1, map_dim_2, ... input_dim). If not specified, will try to figure out the shape from w
        """
        if shape is not None:
            self._shape = shape
            self._map_shape = shape[:-1]
            self._ndim = len(self._map_shape)
            self._input_shape = shape[-1]
            self._size = np.prod(shape[:-1])

            self.w = w
            self._w_shape = w.reshape(shape)
        else:
            # interpret from the SOM w if we can
            self._shape = w.shape
            self._map_shape = w.shape[:-1]
            self._ndim = len(self._map_shape)
            self._input_shape = w.shape[-1]
            self._size = np.prod(w.shape[:-1])

            # flatten w
            self._w_shape = w
            self.w = w.reshape((self._size, self._input_shape))

        assert self.w.shape == (self._size, self._input_shape), "SOM assumes cells flattened to 1d indexing"

    def __getitem__(self, *args, **kwargs):
        return self._w_shape.__getitem__(*args, **kwargs)

    @property
    def ndim(self):
        return self._ndim

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    @property
    def map_shape(self):
        return self._map_shape

    @property
    def input_shape(self):
        return self._input_shape

    @classmethod
    def read(cls, path, name='som'):
        w = pd.read_hdf(path, '{0}/weight'.format(name)).values
        try:
            shape = pd.read_hdf(path, '{0}/shape'.format(name)).values
        except KeyError:
            shape = None

        try:
            # old version
            kind = pd.read_hdf(path, '{0}/type'.format(name))['type'].values[0]
        except TypeError:
            with h5py.File(path, 'r') as h5f:
                kind = h5f['{0}/type'.format(name)][:].tolist()[0]
                kind = kind.decode('utf-8')

        som_class = getattr(sompz, kind)
        som = som_class(w, shape=shape)
        return som

    def write(self, path, name='som'):
        pd.DataFrame(self.w).to_hdf(path, '{0}/weight'.format(name))
        pd.Series(self.shape).to_hdf(path, '{0}/shape'.format(name))
        # pd.DataFrame({'type': [self.__class__.__name__]}).to_hdf(path, '{0}/type'.format(name))
        with h5py.File(path, 'r+') as h5f:
            try:
                h5f.create_dataset('{0}/type'.format(name), data=[self.__class__.__name__.encode('utf-8')])
            except RuntimeError:
                del h5f['{0}/type'.format(name)]
                h5f.create_dataset('{0}/type'.format(name), data=[self.__class__.__name__.encode('utf-8')])

    def copy(self):
        return self.__class__(self.w.copy(), self._shape)

    def __call__(self, x, ivar):
        return self.evaluate(x, ivar)

    def evaluate(self, x, ivar):
        """Return chi2 of input x's and ivar's

        Parameters
        ----------
        x :         Input vector (n_samples, input_shape)
        ivar :      Input inverse variance of x (n_samples, input_shape, input_shape)

        Returns
        -------
        chi2 :      Chi2 fit of each sample to each map cell (n_samples, som_size)

        Notes
        -----
        The input x does not have to have the full span of self.input_shape.
        The chi2 will be evaluated only up to the shape of x. This means that
        you can use a subset of the dimensions as follows: if we trained on
        dimensions griz, then you could pass in gr, but not gi or iz.

        """
        # check number of dims required
        dim = x.shape[1]
        if dim != self.input_shape:
            print('Warning! Trying to evaluate SOM with shape {0} using input of shape {1}. I hope you meant to do this!'.format(self.input_shape, dim))

        n_samples, n_dims = x.shape
        chi2 = np.zeros((n_samples, self.size), dtype=np.float64)
        evaluate_som(x, ivar, self.w, chi2)  # operates on chi2
        return chi2

    def cell1d_to_cell(self, c):
        """Takes 1d assignment vector and turns into Nd

        Parameters
        ----------
        c :     A list of integers (n_samples)

        Returns
        -------
        cND :   Cell assignments (len(map_shape), n_samples)
        """
        cND = np.unravel_index(c, self.map_shape)
        return cND

    def cell_to_cell1d(self, cND):
        """Takes Nd assignment vector and turns into 1d

        Parameters
        ----------
        cND :   Cell assignments (len(map_shape), n_samples)

        Returns
        -------
        c :     A list of integers (n_samples)
        """
        c = np.ravel_multi_index(cND, self.map_shape)
        return c

    def cell1d_to_cell2d(self, c):
        """Takes 1d assignment vector and turns into 2d

        Parameters
        ----------
        c :     A list of integers (n_samples)

        Returns
        -------
        c0, c1: Two dimensional versions of c

        """
        # NOTE: should be exactly same as cell1d_to_cell
        c0 = c % self.map_shape[0]
        c1 = c // self.map_shape[0]
        return c0, c1

    def cell2d_to_cell1d(self, c0, c1):
        """Takes 1d assignment vector and turns into 2d

        Parameters
        ----------
        c0, c1: Two dimensional versions of c

        Returns
        -------
        c :     A list of integers (n_samples)

        """
        # NOTE: should be exactly same as cell_to_cell1d
        c = c1 * self.map_shape[0] + c0
        return c

    def assign(self, x, ivar, verbose=True, diag_ivar=False):
        """Assign sample to a som cell

        Parameters
        ----------
        x :         input data of shape (n_samples, n_dim)
        ivar :      inverse variance of input data (n_samples, n_dim, n_dim)
        verbose :   Print extra information?

        Returns
        -------
        cell :      Best matching cell (by chi2) for each sample (n_samples) in
                    1d coordinate

        """
        # check number of dims required
        dim = x.shape[1]
        if dim != self.input_shape and verbose:
            print('Warning! Trying to assign to SOM with shape {0} using input of shape {1}. I hope you meant to do this!'.format(self.input_shape, dim))

        n_samples = len(x)
        cell = np.zeros(n_samples, dtype=np.int32)
        # cell = assign_bmu(x, ivar, self.w, cell, verbose, diag_ivar)
        assign_bmu(x, ivar, self.w, cell, verbose, diag_ivar)
        return cell    
def train_som(x, ivar, map_shape=[10, 10], learning_rate=0.5, max_iter=2e6, min_val=1e-4, verbose=False, diag_ivar=False, replace=False):
    """Calculate Self Organizing Map

    Parameters
    ----------
    x :             input data of shape (n_samples, n_dim)
    ivar :          inverse variance of input data (n_samples, n_dim, n_dim)
    map_shape :     desired output map shape = [dim1, dim2]. (n_out_dim,)
    learning_rate : float usually between 0 and 1. Sets how large of a
                    change we can effect in the weights at each step by
                    multiplying the change by:
                        learning_rate ** (step_t / total_t)
    max_iter :      maximum number of steps in algorithm fit
    min_val :       minimum parameter difference we worry about in updating
                    SOM. This in practice usually doesn't come up, as we limit
                    the range of cells a SOM may update to be less than one
                    wrap around the map.
    verbose :       Print updates?

    Returns
    -------
    w : self organizing map weights (dim1, dim2, n_dims)

    Notes
    -----
    Suggest whitening your data to span 0-1 range before training

    """

    # initialize w
    # WARNING: initial w is between 0 and 1, not the full range of values in x
    if verbose:
        print('Initializing SOM weight map with shape ({0}, {1})'.format(np.prod(map_shape), x.shape[1]))
    w = np.random.random(size=(np.prod(map_shape), x.shape[1]))
    # select what x and ivar we cycle through
    max_iter = int(max_iter)
    if verbose:
        print('Choosing {0} draws from {1} training samples'.format(max_iter, len(x)))
    choices = build_choices(x, max_iter, replace)

    # update w
    # these arrays cannot be made inside numba functions with nopython=True, so we make them here:
    cND = np.zeros(len(map_shape), dtype=int)
    best_cellND = np.zeros(len(map_shape), dtype=int)
    sigma2_s = np.max(map_shape) ** 2
    update_som_weights(x, ivar, w, choices, cND, best_cellND, learning_rate, sigma2_s, min_val, map_shape, verbose, diag_ivar)

    return w

@numba.jit(nopython=True)  # use nopython to make sure we aren't dropping back to python objects
def update_som_weights(x, ivar, w, choices, cND, best_cellND, learning_rate, sigma2_s, min_val, map_shape, verbose=False, diag_ivar=False):
    """Update Self Organizing Map

    Parameters
    ----------
    x : input data of shape (n_samples, n_dim)
    ivar : inverse variance of input shape (n_samples, n_dim, n_dim)
    w : initial self organizing map weights (w_dims, n_dims)
    choices : indices of samples to draw from x
    learning_rate : float usually between 0 and 1. Sets how large of a
                    change we can effect in the weights at each step by
                    multiplying the change by:
                        learning_rate ** (step_t / total_t)
    min_val : minimum parameter difference we worry about in updating SOM
    map_shape : shape of map w is supposed to be, not including n_dims
    verbose : print updates?

    Returns
    -------
    w : self organizing map weights (dim1, dim2, n_dims)

    Notes
    -----
    n_dims is decided based on x, not w

    """
    n_samples, n_dims = x.shape
    w_dims = w.shape[0]
    map_dims = len(map_shape)
    n_iter = len(choices)

    # sigma2_s = (np.min([j_dims, k_dims])) ** 2
    j_dims = 0
    for i in range(map_dims):
        ji = map_shape[i]
        if ji > j_dims:
            j_dims = ji
    # set the limit in distance the SOM will update
    max_N_diff = np.int(0.5 * j_dims + 1)

    n_print_iter = 10000
    # print(n_samples, n_dims, j_dims, k_dims, n_iter, max_N_diff)

    one_over_n_iter = 1. / n_iter
    one_over_n_dims = 1. / n_dims

    for t in range(n_iter):
        index = choices[t]
        # find index of minimal chi2
        chi2_min, c_min = find_minimum_chi2(x, ivar, w, w_dims, n_dims, one_over_n_dims, index, diag_ivar)
        if c_min == -1:
            raise Exception('Must assign a cell when fitting SOM')

        # if we are only taking one step, then use the values we stuck in
        if t == 0 and n_iter == 1:
            t_ratio = t * one_over_n_iter
            # learning rate
            a = learning_rate
            sigma_neg2_t = sigma2_s ** -1
        else:
            t_ratio = t * one_over_n_iter
            # learning rate
            a = learning_rate ** t_ratio
            sigma_neg2_t = sigma2_s ** (t_ratio - 1)

        # window the indices we update
        N_diff = np.int(np.sqrt(np.abs(np.log(min_val / a) / sigma_neg2_t)))
        if verbose:
            if t % n_print_iter == 0:
                print('step t: ', t, ' . Fraction done: ', t_ratio)
                # print('Distance away we we would update based on min_val and time step: ', N_diff, ' . Max distance away we would ever look : ', max_N_diff)
                # print('t / total steps: ', t_ratio, ' learning rate ** tratio: ', a, ' j_dims ** 2 ** (tratio - 1) :', sigma_neg2_t)
                print('index: ', index, ' current best cell: ', c_min, ' chi2 of best cell: ', chi2_min)
        if N_diff > max_N_diff:
            N_diff = max_N_diff

        if N_diff == 0:
            # we are done, so stop!
            # print('Stopping SOM at Iteration {0}'.format(t))
            print('Stopping SOM because N_diff == 0')
            print('step: ', t)
            return w

        # get j, k, etc of best matching cell
        unravel_index(c_min, map_shape, best_cellND)

        # update all cells
        for c in range(w_dims):
            # convert to ND
            unravel_index(c, map_shape, cND)

            # get distance, including accounting for toroidal topology
            diff2 = 0.0
            for di in range(map_dims):
                best_c = best_cellND[di]
                ci = cND[di]
                i_dims = map_shape[di]
                diff = (ci - best_c)
                while diff < 0:
                    diff += i_dims
                while diff >= i_dims:
                    diff -= i_dims
                diff2 += diff * diff

            # get Hbk
            Hbk = np.exp(-sigma_neg2_t * diff2)

            # update
            for i in range(n_dims):
                w[c, i] += a * Hbk * (x[index, i] - w[c, i])

    # return w

@numba.jit(nopython=True)
def assign_bmu(x, ivar, w, bmu, verbose=True, diag_ivar=False):
    """Assign best matching unit from self organizing map weights

    Parameters
    ----------
    x : input data of shape (n_samples, n_dim)
    ivar : inverse variance of input shape (n_samples, n_dim, n_dim)
    w : self organizing map weights (dim1, dim2, n_dim)

    Returns
    -------
    bmu : best matching cell that each x goes into (n_samples,)

    """

    n_samples, n_dims = x.shape
    w_dims = w.shape[0]

    n_print_iter = 10000
    one_over_n_dims = 1. / n_dims

    for t in range(n_samples):
        if t % n_print_iter == 0:
            if verbose:
                print('assigning sample ', t, ' out of ', n_samples)

        chi2_min, bmu_t = find_minimum_chi2(x, ivar, w, w_dims, n_dims, one_over_n_dims, t, diag_ivar)

        # hooray, now save the best matching unit for input t
        bmu[t] = bmu_t

    # return bmu
@numba.jit(nopython=True)
def find_minimum_chi2(x, ivar, w, w_dims, n_dims, one_over_n_dims, index, diag_ivar=False):
    """Find which cell has the minimum chi2

    Parameters
    ----------
    x : input data of shape (n_samples, n_dims)
    ivar : inverse variance of input shape (n_samples, n_dims, n_dims)
    w : self organizing map weights (w_dims, n_dims)
    w_dims : number of map cells
    n_dims : input vector dims
    index : which object in x we are looking at

    Returns
    -------
    chi2_min : minimum chi2 of best match
    cell_min : index of best match

    """

    chi2_min = 1e100
    cell_min = -1
    # find index of minimal chi2
    for c in range(w_dims):
        chi2 = evaluate_chi2(x, ivar, w, n_dims, one_over_n_dims, index, cell=c, chi2_break=chi2_min, diag_ivar=diag_ivar)
        if chi2 < chi2_min:
            chi2_min = chi2
            cell_min = c

    if cell_min == -1:
        print('No minimum cell found for object:', index)
        print('flux is:', x[index])
        print('ivar is:', ivar[index])

    return chi2_min, cell_min

@numba.jit(nopython=True)
def evaluate_chi2(x, ivar, w, n_dims, one_over_n_dims, index, cell, chi2_break=2e1000, diag_ivar=False):
    """Get chi2. Break if chi2 is larger than chi2_break.
    """
    chi2 = 0.0
    give_up = False
    for i in range(n_dims):
        chi2 += (x[index, i] - w[cell, i]) * (x[index, i] - w[cell, i]) * ivar[index, i, i] * one_over_n_dims
        if chi2 > chi2_break:
            # it only gets worse
            give_up = True
            break
        if not diag_ivar:
            # take advantage of symmetry
            for i2 in range(i + 1, n_dims):
                chi2 += 2 * (x[index, i] - w[cell, i]) * (x[index, i2] - w[cell, i2]) * ivar[index, i, i2] * one_over_n_dims
                if chi2 > chi2_break:
                    # it only gets worse
                    give_up = True
                    break
        if give_up:
            break
    return chi2
