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

def histogram_from_fullpz(df, key, overlap_weighted, bin_edges, full_pz_end = 6.00, full_pz_npts=601):
    # TODO I will need to make this consistent with standards in rail
    '''Preserve bins from Laigle'''
    dz_laigle = full_pz_end / (full_pz_npts - 1)
    condition = np.sum(~np.equal(bin_edges, np.arange(0 - dz_laigle/2.,
                                    full_pz_end + dz_laigle,
                                    dz_laigle)))
    assert condition == 0
    # bin_edges: [-0.005, 0.005], (0.005, 0.015], ... (5.995, 6.005]

    single_cell_hists = np.zeros((len(df), len(key)))

    overlap_weights = np.ones(len(df))
    if(overlap_weighted):
        overlap_weights = df['overlap_weight'].values

    single_cell_hists[:,:] = df[key].values

    # normalize sompz p(z) to have area 1
    dz = 0.01
    area = np.sum(single_cell_hists, axis=1) * dz
    area[area == 0] = 1 # some galaxies have pz with only one non-zero point. set these galaxies' histograms to have area 1
    area = area.reshape(area.shape[0], 1)
    single_cell_hists = single_cell_hists / area

    # response weight normalized p(z)
    single_cell_hists = np.multiply(overlap_weights, single_cell_hists.transpose()).transpose()

    # sum individual galaxy p(z) to single cell p(z)
    hist = np.sum(single_cell_hists, axis=0)

    # renormalize p(z|c)
    area = np.sum(hist) * dz
    hist = hist / area

    return hist