#!/bin/bash
## perlmutter
#SBATCH --qos=regular
#SBATCH --time=47:59:00
#SBATCH --nodes=1
#SBATCH -C cpu  # perlmutter only; invalid on stellar
#SBATCH --error="inform.err"
#SBATCH --output="inform.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmyles@astro.princeton.edu
##stellar
##SBATCH --account=astro
##SBATCH --time=71:59:00
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=64


# perlmutter
module load python
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load PrgEnv-gnu
module load cray-hdf5-parallel

# stellar
#module purge
#module load openmpi/gcc/4.1.6
#module load hdf5/gcc/openmpi-4.1.6/1.14.4
#module load anaconda3/2025.6

#conda activate sompz
#conda activate rail_mpi4py
conda activate rail_sompz

#ceci test_oneinformer_bigfiles_jtm.yml
#ceci test_POSTFIX_fullpipe_mod_cardinal.yml
# if cardinal_inform.yml exists, run it, otherwise run pzdc_inform.yml
if [ -f cardinal_inform.yml ]; then
    ceci cardinal_inform.yml
else
    ceci pzdc_inform.yml
fi
