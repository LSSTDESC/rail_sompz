import pdb
import os
import datetime

import pandas as pd
import matplotlib.pyplot as plt

import optax
import pzflow
from pzflow import Flow
from pzflow.bijectors import Chain, ShiftBounds, RollingSplineCoupling
from pzflow.distributions import Uniform

import jax.numpy as jnp

#datadir_root = '/Users/jmyles/data/'
#datadir_root = '/pscratch/sd/j/jmyles/'
datadir_root = '/home/jm8767/data/'
datadir = os.path.join(datadir_root, 'sompz_buzzard/2024-06-17/')
#datestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # '2024-06-18_16-03-13'
datestr = '2024-06-18_18-39-47'
outdir = os.path.join(datadir, datestr)
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
# load same deep catalog used for SOM training
infile = os.path.join(datadir, 'balrog_data_subcatalog.h5') # 'Chinchilla-3_v4_LSST_0_spec.h5'
balrog_data = pd.read_hdf(infile)
spec_data = balrog_data[balrog_data['REDSHIFT FIELD'] == True]
del balrog_data

#infile = os.path.join(datadir, 'Chinchilla-3_v4_LSST_0_spec.h5')
#spec_data = pd.read_hdf(infile)

nsamp = 1_000_000
#spec_data = spec_data.sample(nsamp)
print(f'spec_data before cut {len(spec_data):,}')
print(spec_data.isnull().values.any())
#spec_data = spec_data.sample(nsamp)
#print(f'spec_data after cut {len(spec_data):,}')

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
batch_size=64 # default 1024
print(f'Using batch_size {batch_size}')

# get minima and maxima for each column
mins = jnp.array(spec_data[NFbands].min(axis=0))
maxs = jnp.array(spec_data[NFbands].max(axis=0))
print(f'Mins {mins}')
print(f'Maxs {maxs}')

# get the number of dimensions
ndim = spec_data[NFbands].shape[1]
print(f'ndim {ndim}')
# build the bijector
bijector = Chain(
    ShiftBounds(mins, maxs, B=4),
    RollingSplineCoupling(ndim, B=5),
)

# set the latent space
latent = Uniform(input_dim=ndim, B=5)

# construct (or load) NF
outfile_flow = os.path.join(outdir, 'pz_c.pkl')
outfile_loss = os.path.join(outdir, 'loss.npy')
if not os.path.exists(outfile_flow):
    flow = Flow(NFbands, bijector=bijector, latent=latent,)
    # train NF to learn p(z|c)
    print(f'Begin training {datetime.datetime.now()}')
    losses = flow.train(spec_data,
                        batch_size=batch_size, optimizer=opt, epochs=200,
                        verbose=True)
    print(f'End training {datetime.datetime.now()}')
    flow.save(outfile_flow)
    jnp.save(outfile_loss, losses)
else:
    print(f'Loading normalizing flow')
    flow = Flow(file=outfile_flow)
    losses = jnp.load(outfile_loss)
    
# plot loss
plt.figure(figsize=(6,8))
plt.plot(losses, '.', label='loss')
plt.ylim((-10, 30))
badvals = jnp.where(losses > 100)[0]
for i, badval in enumerate(badvals):
    plt.axvline(badval, 0.95, 1, alpha=0.25, color='k', label='loss > 100' if i == 0 else '')
plt.xlabel("Epoch")
plt.ylabel("Training loss")
outfile_fig = os.path.join(outdir, 'loss.png')
plt.savefig(outfile_fig)
plt.close()

# sample p(z|c)
samples = flow.sample(nsamp, seed=0)

# plot samples
band_idx = 3
plt.figure()
plt.hist2d(samples["Z"], samples[deepbands[band_idx]], bins=200)
plt.xlabel("Z")
plt.ylabel(deepbands[band_idx])
outfile_fig = os.path.join(outdir, 'samples.png')
plt.savefig(outfile_fig)
plt.close()

#pdb.set_trace()
# generate PDFs in grid
grid = jnp.linspace(0, 3, 100)
pdfs = flow.posterior(spec_data, column="Z", grid=grid)
outfile_pdfs = os.path.join(outdir, 'pdfs.npz')
jnp.save(outfile_pdfs, pdfs)

# plot PDFs
plt.figure()
plt.plot(grid, pdfs[0])
plt.title(f"${deepbands[band_idx]}$ = {spec_data[deepbands[band_idx]].iloc[0]:.2f}")
#plt.title(f"$y$ = {data['y'][0]:.2f}")
plt.xlabel("$Z$")
plt.ylabel(f"$p(Z|{deepbands[band_idx]})$")
outfile_fig = os.path.join(outdir, 'pdf.png')
plt.savefig(outfile_fig)
plt.close()
