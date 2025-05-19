""" Utility functions """

import os
from rail.utils.catalog_utils import CatalogConfigBase

from rail import sompz

import numpy as np

RAIL_SOMPZ_DIR = os.path.abspath(os.path.join(os.path.dirname(sompz.__file__), '..', '..'))

def mag2flux(mag, zero_pt=30):  # pragma: no cover
    # zeropoint: M = 30 <=> f = 1
    exponent = (mag - zero_pt)/(-2.5)
    val = 1 * 10 ** (exponent)
    return val

def flux2mag(flux, zero_pt=30):  # pragma: no cover
    return zero_pt - 2.5 * np.log10(flux)

def fluxerr2magerr(flux, fluxerr):  # pragma: no cover
    coef = -2.5 / np.log(10)
    return np.abs(coef * (fluxerr / flux))

def magerr2fluxerr(magerr, flux):  # pragma: no cover
    coef = np.log(10) / -2.5
    return np.abs(coef * magerr * flux)

def luptize(flux, var, s, zp):  # pragma: no cover
    # s: measurement error (variance) of the flux (with zero pt zp) of an object at the limiting magnitude of the survey
    # a: Pogson's ratio
    # b: softening parameter that sets the scale of transition between linear and log behavior of the luptitudes
    a = 2.5 * np.log10(np.exp(1))
    b = a**(1./2) * s
    mu0 = zp -2.5 * np.log10(b)

    # turn into luptitudes and their errors
    lupt = mu0 - a * np.arcsinh(flux / (2 * b))
    lupt_var = a ** 2 * var / ((2 * b) ** 2 + flux ** 2)
    return lupt, lupt_var

def mean_of_hist(y, bins):  # pragma: no cover # n=1
    """Given a histogram and its bins return the mean
    of the distribution.
    Parameters
    ----------
    y :     A histogram of values
    bins :  The bins of the histogram
    Returns
    -------
    normalization, mean, sigma
    """
    dx = np.diff(bins)
    x = 0.5 * (bins[1:] + bins[:-1])
    normalization = np.trapz(y, x=x, dx=dx)
    mean = np.trapz(x * y, x=x, dx=dx) / normalization
    result = mean
    #var = np.trapz((x - mean) ** 2 * y, x=x, dx=dx) / normalization
    #sigma = np.sqrt(var)
    return result

def selection_wl_cardinal(mag_i, mag_r, mag_r_limit, size,
                          psf_r=0.9, imag_max=25.1):  # pragma: no cover
    select_mag_i = mag_i < imag_max
    select_mag_r = mag_r < -2.5 * np.log10(0.5) + mag_r_limit
    select_psf_r = np.sqrt(size**2  + (0.13 * psf_r)**2) > 0.1625 * psf_r

    select = select_mag_i & select_mag_r & select_psf_r

    return select


class SompzWideTestCatalogConfig(CatalogConfigBase):
    """Configuration for SOMPZ test data wide field"""

    tag = "som_pz_wide"
    hdf5_groupname = ""
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [26.4, 27.8, 27.1, 26.7, 25.8, 24.6]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "{band}"
    band_err_template = "{band}_err"
    filter_file_template = "DC2LSST_{band}"
    ref_band = "i"
    redshift_col = "redshift"


class SompzDeepTestCatalogConfig(CatalogConfigBase):
    """Configuration for SOMPZ test data deep field"""

    tag = "som_pz_deep"
    hdf5_groupname = ""
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'F']
    maglims = [26.4, 27.8, 27.1, 26.7, 25.8, 24.6, 27.8, 28.1, 27.5]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31, 0.68, 0.60, 0.47]
    lsst_err_band_replace = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    zp_errors = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    band_template = "{band}"
    band_err_template = "{band}_err"
    filter_file_template = "DC2LSST_{band}"
    ref_band = "i"
    redshift_col = "redshift"
