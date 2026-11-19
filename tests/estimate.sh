#!/bin/bash
##SBATCH --qos=debug # perlmutter
##SBATCH --time=00:30:00 # perlmutter
##SBATCH --qos=regular # debug # regular
#SBATCH --time=47:59:00 # both
#SBATCH --nodes=1 # both
##SBATCH --ntasks=32 # perlmutter: match nprocess in cardinal_estimate.yml estimator stages
#SBATCH --ntasks=96 # stellar (depends on type of node assigned -- 96 is safe but often wasteful)
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

# add /usr/bin/latex executable to PATH for matplotlib to use


#### conda activate sompz
#### conda activate rail_mpi4py
conda activate rail_sompz 
export PATH=$PATH:/usr/bin

# identify directory with catalogs
DATADIR_CATALOG=$(python $HOME/repositories/lsst-y1-nz-study/get_run_catalog_datadir.py)

# build list of directories to run estimate stages in
CWD=$(pwd)
DATADIRS_RUN=()
# if realizations found, do estimate analysis for run-v*/realization_*_v3
for dir in $(ls -d realization_*_v3 2>/dev/null | sort -V); do
    if [ -d "$dir" ]; then
        DATADIRS_RUN+=("$CWD/${dir%/}")
    fi
done
# elif no realizations found, do estimate analysis for run-v*/ directory itself
# if [ ${#DATADIRS_RUN[@]} -eq 0 ]; then
    # DATADIRS_RUN+=("$CWD")
# fi

# do estimate analysis for run-v*/ directory itself
DATADIRS_RUN+=("$CWD")

echo "DATADIRS_RUN: ${DATADIRS_RUN[@]}"
echo $DATADIR_CATALOG
#echo "Running estimate for $DATADIRS_RUN"

for DATADIR_RUN in "${DATADIRS_RUN[@]}"; do
    echo "Doing estimate stages for $DATADIR_RUN"
    cd "$DATADIR_RUN" || exit 1

    if [ -f cardinal_estimate.yml ]; then
        ceci cardinal_estimate.yml
    elif [ -f estimate.yml ]; then
        ceci estimate.yml
    else
        ceci nzdc_estimate.yml
    fi

    python $HOME/repositories/lsst-y1-nz-study/make_plots_for_sompz_runs.py --datadirs-runs "$DATADIR_RUN" --datadirs-catalogs "$DATADIR_CATALOG"

    cd "$CWD" || exit 1
done
