#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --error="testscript_inform_big.err"
#SBATCH --output="testscript_inform_big.out"

module load python
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load PrgEnv-gnu
module load cray-hdf5-parallel
#conda activate sompz
#conda activate rail_mpi4py
conda activate rail_sompz

#ceci test_oneinformer_bigfiles_jtm.yml
#ceci test_POSTFIX_fullpipe_mod_cardinal.yml
ceci cardinal_inform.yml
