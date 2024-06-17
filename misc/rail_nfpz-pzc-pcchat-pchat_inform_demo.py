import os
import datetime

import pandas as pd
import matplotlib.pyplot as plt

import pzflow
import optax
from pzflow import Flow
import jax.numpy as jnp

#datadir_root = '/Users/jmyles/data/'
#datadir_root = '/pscratch/sd/j/jmyles/'
datadir_root = '/home/jm8767/data/'
datadir = os.path.join(datadir_root, 'sompz_buzzard/2024-06-17/')
outdir = datadir
infile = os.path.join(datadir, 'Chinchilla-3_v4_LSST_0_spec.h5')
spec_data = pd.read_hdf(infile)

nsamp = 100_000
#spec_data = spec_data.sample(nsamp)
print(f'spec_data before cut {len(spec_data):,}')
print(spec_data.isnull().values.any())
spec_data = spec_data.sample(nsamp)
print(f'spec_data after cut {len(spec_data):,}')

# set training hyperparameters
# TODO: change lines below to read from config file
bands_deep = ['lsst_u', 'lsst_g', 'lsst_r', 'lsst_i', 'lsst_z', 
              'VISTA_Filters_at80K_forETC_Y',
              'VISTA_Filters_at80K_forETC_J',
              'VISTA_Filters_at80K_forETC_H',
              'VISTA_Filters_at80K_forETC_Ks',] 
deepbands = [f'TRUEMAG_{band}' for band in bands_deep] # [:5]
NFbands = ["Z"] + deepbands

opt = optax.adam(learning_rate=1e-6) # default 1e-3
batch_size=64
print(f'Using batch_size {batch_size}')

# train NF to learn p(z|c)
flow = Flow(NFbands)
print(f'Begin training {datetime.datetime.now()}')
losses = flow.train(spec_data, batch_size=batch_size, optimizer=opt, verbose=True)
print(f'End training {datetime.datetime.now()}')
outfile = os.path.join(outdir, 'pz_c.pkl')
flow.save(outfile)

# plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
outfile_fig = os.path.join(outdir, 'loss.png')
outfile_data = os.path.join(outdir, 'loss.npz')
plt.savefig(outfile_fig)
jnp.save(outfile_data, losses)

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
grid = jnp.linspace(0, 3, 100)
pdfs = flow.posterior(spec_data, column="Z", grid=grid)
outfile_pdfs = os.path.join(outdir, 'pdfs.npz')
jnp.save(outfile_pdfs, pdfs)

# plot PDFs
plt.plot(grid, pdfs[0])
plt.title(f"${deepbands[band_idx]}$ = {spec_data[deepbands[band_idx]].iloc[0]:.2f}")
#plt.title(f"$y$ = {data['y'][0]:.2f}")
plt.xlabel("$Z$")
plt.ylabel(f"$p(Z|{deepbands[band_idx]})$")
outfile_fig = os.path.join(outdir, 'pdf.png')
plt.savefig(outfile_fig)
