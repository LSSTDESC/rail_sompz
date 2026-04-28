#!/bin/bash
##SBATCH --qos=debug # perlmutter
##SBATCH --time=00:30:00 # perlmutter
##SBATCH --qos=regular # debug # regular
#SBATCH --time=48:00:00 # both
#SBATCH --nodes=1 # both
##SBATCH --ntasks=32 # perlmutter: match nprocess in cardinal_estimate.yml estimator stages
#SBATCH --ntasks=96 # stellar: match nprocess in cardinal_estimate.yml
##SBATCH --cpus-per-task=1 # unknown
##SBATCH -C cpu # perlmutter only; invalid on stellar
#SBATCH --error="estimate.err" # both
#SBATCH --output="estimate.out" #both
#SBATCH --mail-type=ALL # both
#SBATCH --mail-user=jmyles@astro.princeton.edu # both
#SBATCH --cpus-per-task=1 # stellar: 1 CPU per MPI process

# perlmutter
# module load python
# module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
# module load PrgEnv-gnu
# module load cray-hdf5-parallel
# module load libfabric/1.20.1 # added 2025-08-22 following appearance of msg below after maintenance
# The following have been reloaded with a version change:
#  1) libfabric/1.20.1 => libfabric/1.22.0

# stellar
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
module purge
#module load intel/2024.2
#module load intel-mpi/intel/2021.7.0
#module load intel-mpi/intel/2021.13
##module load hdf5/gcc/openmpi-4.1.6/1.14.4
#module load hdf5/intel-2021.1/openmpi-4.1.0/1.10.6
module load openmpi/gcc/4.1.6
module load hdf5/gcc/openmpi-4.1.6/1.14.4
module load anaconda3/2025.6


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

# if cardinal_estimate.yml exists, run it, otherwise run pzdc_estimate.yml
# if [ -f cardinal_estimate.yml ]; then
#     ceci cardinal_estimate.yml
# else
#     ceci pzdc_estimate.yml
# fi

#python $HOME/repositories/lsst-y1-nz-study/fit-smail-dist-to-rail-sompz-nz.py ./
python $HOME/repositories/lsst-y1-nz-study/make-plots-for-sompz-runs.py --datadirs-runs $DATADIR_RUN --datadirs-catalogs $DATADIR_CATALOG