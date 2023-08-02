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
