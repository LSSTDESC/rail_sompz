import os
import pzflow

import pandas as pd
import matplotlib.pyplot as plt

from pzflow import Flow
import jax.numpy as jnp

#datadir = '/Users/jmyles/data/sompz_buzzard/2024-06-14'
datadir = '/pscratch/sd/j/jmyles/sompz_buzzard/2024-06-14/'
outdir = datadir
infile = os.path.join(datadir, 'Chinchilla-3_v4_LSST_0_spec.h5')
spec_data = pd.read_hdf(infile)

nsamp = 100_000
#spec_data = spec_data.sample(nsamp)
print(f'{len(spec_data):,}')

# TODO: change lines below to read from config file
bands_deep = ['lsst_u', 'lsst_g', 'lsst_r', 'lsst_i', 'lsst_z', 
              'VISTA_Filters_at80K_forETC_Y', 'VISTA_Filters_at80K_forETC_J', 'VISTA_Filters_at80K_forETC_H', 'VISTA_Filters_at80K_forETC_Ks',] 
deepbands = [f'TRUEMAG_{band}' for band in bands_deep] # [:5]
NFbands = ["Z"] + deepbands

# train NF to learn p(z|c)
flow = Flow(NFbands)
losses = flow.train(spec_data, verbose=True)
flow.save('pz_c.pkl')

# plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
outfile_fig = os.path.join(outdir, 'loss.png')
outfile_data = os.path.join(outdir, 'loss.npz')
plt.savefig(outfile_fig)
np.save(outfile_data, losses)

# sample p(z|c)
samples = flow.sample(nsamp, seed=0)

# plot samples
band_idx = 1
plt.hist2d(samples["Z"], samples[deepbands[band_idx]], bins=200)
plt.xlabel("Z")
plt.ylabel(deepbands[band_idx])
outfile_fig = os.path.join(outdir, 'samples.png')
plt.savefig(outfile_fig)

# generate PDFs in grid
grid = jnp.linspace(-2, 2, 100)
pdfs = flow.posterior(spec_data, column="Z", grid=grid)

# plot PDFs
plt.plot(grid, pdfs[0])
plt.title(f"${deepbands[band_idx]}$ = {spec_data[deepbands[band_idx]].iloc[0]:.2f}")
#plt.title(f"$y$ = {data['y'][0]:.2f}")
plt.xlabel("$Z$")
plt.ylabel(f"$p(Z|{deepbands[band_idx]})$")
outfile_fig = os.path.join(outdir, 'pdf.png')
plt.savefig(outfile_fig)
