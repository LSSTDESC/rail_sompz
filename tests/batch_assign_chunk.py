# usual imports
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from rail.core.utils import RAILDIR
from rail.core import RailStage
from rail.core.data import TableHandle
import qp
import ceci
from mpi4py import MPI
from schwimmbad import MPIPool
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
import rail
import yaml
import argparse
# change to your rail location
from rail.estimation.algos.sompz import SOMPZEstimator,SOMPZEstimatorBase, SOMPZEstimatorWide, SOMPZEstimatorDeep,SOMPZTomobin,SOMPZPzc, SOMPZPzchat, SOMPZnz, SOMPZPc_chat,Hdf5Handle
import pandas as pd 
import gc
import h5py
import tables_io
DS = RailStage.data_store
DS.__class__.allow_overwrite = True
import ceci


parser = argparse.ArgumentParser(description="This package assign each galaxy into a SOM cell",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("yml", help="yamlfile", type=str)
args = parser.parse_args()
with open(args.yml, 'r') as file:
    yamlfile = yaml.safe_load(file)


outdir = yamlfile['outpath']
if not os.path.isdir(outdir):
    os.makedirs(outdir)

if rank == 0:
        balrog_data= tables_io.read(outdir+os.path.basename(yamlfile['balrog_data'])+f"_{int(eval(yamlfile['NbalrogsubsampleN']))}.hdf5") 
        zall = balrog_data[yamlfile['specz_name']][~np.isnan(balrog_data[yamlfile['specz_name']])]
        del balrog_data
        quantile = np.linspace(0,1,10)
        bin_edges = np.quantile(zall, quantile)
        del zall
        gc.collect()
else:
        bin_edges = None
bin_edges = comm.bcast(bin_edges, root=0)
balrog_data = outdir+os.path.basename(yamlfile['balrog_data'])+f"_{int(eval(yamlfile['NbalrogsubsampleN']))}.hdf5"
wide_data= outdir+os.path.basename(yamlfile['wide_data'])+f"_{int(eval(yamlfile['NwidesubsampleN']))}.hdf5"
spec_data= outdir+os.path.basename(yamlfile['spec_data'])+f"_{int(eval(yamlfile['NspecsubsampleN']))}.hdf5"

balrog_data_in = Hdf5Handle('data', path=balrog_data).read()
wide_data_in = Hdf5Handle('data', path=wide_data).read()
spec_data_in = Hdf5Handle('data', path=spec_data).read()

zall = balrog_data_in[yamlfile['specz_name']][~np.isnan(balrog_data_in[yamlfile['specz_name']])]
quantile = np.linspace(0,1,10)
bin_edges = np.quantile(zall, quantile)
del zall
gc.collect()



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

deep_som_params = dict(inputs=deepbands, input_errs=deeperrs,
                  zero_points=zeropts,
                  convert_to_flux=False, 
                  set_threshold=True, thresh_val=1.e-5, 
                  som_minerror=1E-3,
                  som_take_log=True, som_wrap=False,som_shape=(yamlfile['deep_som_size'],yamlfile['deep_som_size']), specz_name=yamlfile['specz_name'],debug=False, bin_edges=bin_edges, model=outdir+yamlfile['deepmodel'].format(yamlfile['deep_som_size']), resume=True)

wide_som_params = dict(inputs=widebands, input_errs=wideerrs,
                  zero_points=zeropts,
                  convert_to_flux=False, 
                  set_threshold=True, thresh_val=1.e-5, 
                  som_minerror=1E-3,
                  som_take_log=True, som_wrap=False,som_shape=(yamlfile['wide_som_size'],yamlfile['wide_som_size']), specz_name=yamlfile['specz_name'],debug=False, bin_edges=bin_edges, model=outdir+yamlfile['widemodel'].format(yamlfile['wide_som_size']), resume=True)




#samples = [spec_data, balrog_data, wide_data]
samples = [spec_data_in, balrog_data_in, wide_data_in]
labels = ['spec_data', 'balrog_data', 'wide_data']

for i, (data, label) in enumerate(zip(samples, labels)):
    
    som_estimate_wide = SOMPZEstimatorWide.make_stage(name="sompz_estimator_wide_"+label,hdf5_groupname="",
                                 assignment= yamlfile['outpath']+label+"_wide_assignment.hdf5", data_path=yamlfile['outpath'], comm=comm, chunk_size=100, 
                                 **wide_som_params)

    som_estimate_deep = SOMPZEstimatorDeep.make_stage(name="sompz_estimator_deep_"+label,hdf5_groupname="",
                                  assignment= yamlfile['outpath']+label+"_deep_assignment.hdf5", 
                                  data_path=yamlfile['outpath'], comm=comm, chunk_size=100,**deep_som_params)
    if i<2:
        # check if deep assignment has been stored to disk
        if not os.path.isfile(yamlfile['outpath']+label+"_deep_assignment.hdf5"):
            #pdb.set_trace()
            _ = som_estimate_deep.estimate(data)
        comm.Barrier()
    # check if wide assignment has been stored to disk
    if not os.path.isfile(yamlfile['outpath']+label+"_wide_assignment.hdf5"):
        _ = som_estimate_wide.estimate(data)
    comm.Barrier()  
    if rank==0:
        if i==0:
            cell_deep_spec_data = outdir + "spec_data_deep_assignment.hdf5"
            cell_wide_spec_data  = outdir + "spec_data_wide_assignment.hdf5"

            sOMPZPzc = SOMPZPzc.make_stage(name="sompz_pzc", specz_name='z', pz_c = outdir+"pz_c.hdf5",
                                           data_path=outdir, bin_edges=bin_edges,
                                           deep_groupname=yamlfile['deep_groupname'],
                                           spec_data=spec_data,
                                           cell_deep_spec_data=cell_deep_spec_data,
                                           cell_wide_spec_data=cell_wide_spec_data)
            #pdb.set_trace()
            sOMPZPzc.estimate()
        if i==1:
            cell_deep_balrog_data = outdir + "balrog_data_deep_assignment.hdf5"
            cell_wide_balrog_data = outdir + "balrog_data_wide_assignment.hdf5"
            sOMPZPc_chat = SOMPZPc_chat.make_stage(specz_name='z',name="sompz_pcchat",deep_som_size=yamlfile['deep_som_size']**2, wide_som_size=yamlfile['wide_som_size']**2, pc_chat = outdir+"pc_chat.hdf5", data_path=outdir, cell_deep_balrog_data=cell_deep_balrog_data,cell_wide_balrog_data=cell_wide_balrog_data)
            sOMPZPc_chat.estimate()

        if i==2:

                cell_wide_wide_data = outdir + "wide_data_wide_assignment.hdf5"
                sOMPZPzchat = SOMPZPzchat.make_stage(name="sompz_pzchat", pz_chat = outdir+"pz_chat.hdf5", data_path=outdir, bin_edges=bin_edges, spec_data=spec_data, cell_deep_spec_data=cell_deep_spec_data, pz_c=outdir+"pz_c.hdf5", pc_chat = outdir+"pc_chat.hdf5",cell_wide_wide_data=cell_wide_wide_data,specz_name='z')
                sOMPZPzchat.estimate()
                sOMPZTomobin = SOMPZTomobin.make_stage(name="sompz_tomobin",  data_path=outdir, bin_edges=bin_edges, spec_data=spec_data, cell_deep_spec_data=cell_deep_spec_data, cell_wide_spec_data=cell_wide_spec_data, tomo_bins_wide= outdir+"tomo_bins_wide.hdf5", specz_name='z')
                sOMPZTomobin.estimate()
                sOMPZnz = SOMPZnz.make_stage(name="sompznz", pz_chat = outdir+"pz_chat.hdf5", data_path=outdir, bin_edges=bin_edges, spec_data=spec_data, cell_deep_spec_data=cell_deep_spec_data, pz_c=outdir+"pz_c.hdf5", pc_chat = outdir+"pc_chat.hdf5",cell_wide_wide_data=cell_wide_wide_data,specz_name='z', tomo_bins_wide=outdir+"tomo_bins_wide.hdf5",nz = outdir+"nz_sompz_estimator.hdf5")
                sOMPZnz.estimate()






