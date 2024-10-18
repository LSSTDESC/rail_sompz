# usual imports
import os
import tables_io
import numpy as np
import matplotlib.pyplot as plt
from rail.core.utils import RAILDIR
from rail.core import RailStage
from rail.core.data import TableHandle
import qp
import ceci
import rail
import yaml
import argparse
# change to your rail location
from rail.estimation.algos.sompz import SOMPZEstimator
import pandas as pd 
import gc
import h5py
import os
import numpy as np
#from rail.core.utils import RAILDIR
import matplotlib.pyplot as plt
import rail
from rail.estimation.algos.sompz import SOMPZInformer,CatInformer, SHARED_PARAMS
from rail.core.data import TableHandle
from rail.core.stage import RailStage
DS = RailStage.data_store
DS.__class__.allow_overwrite = True
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="This package assign each galaxy into a SOM cell",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("yml", help="yamlfile", type=str)
parser.add_argument('--stop', type=str2bool, help='Early stop', default=False)
args = parser.parse_args()
with open(args.yml, 'r') as file:
    yamlfile = yaml.safe_load(file)

outdir = yamlfile['outpath']
if not os.path.isdir(outdir):
    os.makedirs(outdir)
if os.path.isfile(outdir+os.path.basename(yamlfile['spec_data'])+f"_{int(eval(yamlfile['NspecsubsampleN']))}.hdf5"):
    balrog_data= tables_io.read(outdir+os.path.basename(yamlfile['balrog_data'])+f"_{int(eval(yamlfile['NbalrogsubsampleN']))}.hdf5") 
    wide_data= tables_io.read(outdir+os.path.basename(yamlfile['wide_data'])+f"_{int(eval(yamlfile['NwidesubsampleN']))}.hdf5")
    spec_data= tables_io.read(outdir+os.path.basename(yamlfile['spec_data'])+f"_{int(eval(yamlfile['NspecsubsampleN']))}.hdf5")
else:
    balrog_data= pd.read_hdf(yamlfile['balrog_data']) 
    balrog_data = balrog_data[(22.5-2.5*np.log10(balrog_data['FLUX_H'])<(yamlfile['Hcut']))&(balrog_data['FLUX_H']/balrog_data['FLUX_ERR_H']>yamlfile['SNRcut'])]
    spec_data = balrog_data[balrog_data[yamlfile['specz_cond']]>0]
    if 'NspecsubsampleN' in yamlfile.keys():
        if int(eval(yamlfile['NspecsubsampleN']))<len(spec_data):
            spec_data = spec_data.sample(int(eval(yamlfile['NspecsubsampleN'])), random_state=1)
        else:
            print(f"spec data is {spec_data} which is smaller than {int(eval(yamlfile['NspecsubsampleN']))}\n", flush=True)
        spec_data = tables_io.convert(spec_data, tables_io.types.NUMPY_HDF5)
        tables_io.write(spec_data, outdir+os.path.basename(yamlfile['spec_data'])+f"_{int(eval(yamlfile['NspecsubsampleN']))}.hdf5")
    if 'NbalrogsubsampleN' in yamlfile.keys():
        if int(eval(yamlfile['NbalrogsubsampleN']))<len(balrog_data):
            balrog_data=balrog_data.sample(int(eval(yamlfile['NbalrogsubsampleN'])), random_state=1)
        else:
            print("balrog data legnth", len(balrog_data), flush=True)
        balrog_data = tables_io.convert(balrog_data, tables_io.types.NUMPY_HDF5)
        tables_io.write(balrog_data, outdir+os.path.basename(yamlfile['balrog_data'])+f"_{int(eval(yamlfile['NbalrogsubsampleN']))}.hdf5")
    gc.collect()

    wide_data= pd.read_hdf(yamlfile['wide_data'])
    wide_data = wide_data[(22.5-2.5*np.log10(wide_data['FLUX_H'])<(yamlfile['Hcut']))&(wide_data['FLUX_H']/wide_data['FLUX_ERR_H']>yamlfile['SNRcut'])]
    if 'NwidesubsampleN' in yamlfile.keys():
        if int(eval(yamlfile['NwidesubsampleN']))<len(wide_data):
            wide_data = wide_data.sample(int(eval(yamlfile['NwidesubsampleN'])), random_state=1)
        else:
            print("Wide data length", len(wide_data), flush=True)
        wide_data = tables_io.convert(wide_data, tables_io.types.NUMPY_HDF5)
        tables_io.write(wide_data, outdir+os.path.basename(yamlfile['wide_data'])+f"_{int(eval(yamlfile['NwidesubsampleN']))}.hdf5")
if args.stop:
    assert(0)

bands = yamlfile['deep_bands']
deepbands = []
deeperrs = []
zeropts = []
for band in bands:
    deepbands.append(f'FLUX_TRUE_{band}')
    deeperrs.append(f'FLUX_TRUE_ERR_{band}')
    zeropts.append(22.5)

bands = yamlfile['wide_bands']
widebands = []
wideerrs = []  
for band in bands:
    widebands.append(f'FLUX_{band}')
    wideerrs.append(f'FLUX_ERR_{band}')
refband_deep = yamlfile['refdeep']
refband_wide = yamlfile['redwide']

som_params_deep = dict(inputs=deepbands, input_errs=deeperrs,
                  zero_points=zeropts,
                  convert_to_flux=False, 
                  set_threshold=True, thresh_val=1.e-5, thresh_val_err=1.0E-5, som_minerror=1E-4, 
                  som_take_log=True, som_wrap=False, som_shape=(yamlfile['deep_som_size'], yamlfile['deep_som_size']), specz_name=yamlfile['specz_name'],debug=False)

som_params_wide = dict(inputs=widebands, input_errs=wideerrs,
                  zero_points=zeropts,
                  convert_to_flux=False,
                  set_threshold=True, thresh_val=1.e-5, thresh_val_err=1.0E-5, som_minerror=1E-4, 
                  som_take_log=True, som_wrap=False, som_shape=(yamlfile['wide_som_size'], yamlfile['wide_som_size']), specz_name=yamlfile['specz_name'],debug=False)

from multiprocessing import Pool 
import multiprocessing
nprocess=yamlfile['nprocess']
print("total process", nprocess, flush=True)
with Pool(nprocess) as p:
        if not os.path.isfile(outdir+yamlfile['deepmodel'].format(yamlfile['deep_som_size'])):
            som_inform = SOMPZInformer.make_stage(name="som_informer", 
                                          groupname="", 
                                          model=outdir+yamlfile['deepmodel'].format(yamlfile['deep_som_size']),pool=[p,nprocess], **som_params_deep)
            som_inform.inform(balrog_data)
        print("deep done", flush=True)
        if not os.path.isfile(outdir+yamlfile['widemodel'].format(yamlfile['wide_som_size'])):
            som_inform = SOMPZInformer.make_stage(name="som_informer", 
                                          groupname="",
                                          model=outdir+yamlfile['widemodel'].format(yamlfile['wide_som_size']),pool=[p,nprocess], **som_params_wide)
            som_inform.inform(wide_data)



