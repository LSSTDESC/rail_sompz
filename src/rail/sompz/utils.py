""" Utility functions """

import os
from rail import sompz

RAIL_SOMPZ_DIR = os.path.abspath(os.path.join(os.path.dirname(sompz.__file__), '..', '..'))

def mag2flux(mag, zero_pt=30):
    # zeropoint: M = 30 <=> f = 1
    exponent = (mag - zero_pt)/(-2.5)
    val = 1 * 10 ** (exponent)
    return val

def flux2mag(flux, zero_pt=30):
    return zero_pt - 2.5 * np.log10(flux)

def fluxerr2magerr(flux, fluxerr):
    coef = -2.5 / np.log(10)
    return np.abs(coef * (fluxerr / flux))

def magerr2fluxerr(magerr, flux):
    coef = np.log(10) / -2.5
    return np.abs(coef * magerr * flux)

def luptize(flux, var, s, zp):
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

def selection_wl_cardinal(mag_i, mag_r, mag_r_limit, size, psf_r=0.9, imag_max=25.1):
    select_mag_i = mag_i < imag_max
    select_mag_r = mag_r < -2.5 * np.log10(0.5) + mag_r_limit
    select_psf_r = np.sqrt(size**2  + (0.13 * psf_r)**2) > 0.1625 * psf_r

    select = select_mag_i & select_mag_r & select_psf_r

    return select
