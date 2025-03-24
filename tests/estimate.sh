#!/bin/bash
#SBATCH --qos=debug # regular
#SBATCH --time=00:29:00
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --error="testscript_estimate_big.err"
#SBATCH --output="testscript_estimate_big.out"

module load python
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load PrgEnv-gnu
module load cray-hdf5-parallel
#conda activate sompz
#conda activate rail_mpi4py
conda activate rail_sompz

#ceci test_pipe_FULL_coriparallel_big.yml
#ceci test_POSTFIX_fullpipe_mod_cardinal.yml
ceci cardinal_estimate.yml
