#!/bin/bash
##SBATCH --qos=debug 
##SBATCH --time=00:30:00
#SBATCH --qos=regular # debug # regular
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --error="estimate.err"
#SBATCH --output="estimate.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmyles@astro.princeton.edu

module load python
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load PrgEnv-gnu
module load cray-hdf5-parallel

# module load libfabric/1.20.1 # added 2025-08-22 following appearance of msg below after maintenance
# The following have been reloaded with a version change:
#  1) libfabric/1.20.1 => libfabric/1.22.0

#### conda activate sompz
#### conda activate rail_mpi4py
conda activate rail_sompz 

#ceci test_pipe_FULL_coriparallel_big.yml
#ceci test_POSTFIX_fullpipe_mod_cardinal.yml
#ceci $HOME/repositories/rail_sompz/tests/cardinal_estimate.yml

DATADIR_RUN=$(pwd)
DATADIR_CATALOG=$(python $HOME/repositories/lsst-y1-nz-study/get_run_catalog_datadir.py)

echo $DATADIR_RUN
echo $DATADIR_CATALOG

ceci cardinal_estimate.yml
python $HOME/repositories/lsst-y1-nz-study/fit-smail-dist-to-rail-sompz-nz.py ./
python $HOME/repositories/lsst-y1-nz-study/make-plots-for-sompz-runs.py --datadirs-runs $DATADIR_RUN --datadirs-catalogs $DATADIR_CATALOG