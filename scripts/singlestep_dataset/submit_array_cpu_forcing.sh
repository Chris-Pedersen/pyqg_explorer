#!/bin/bash

#SBATCH --job-name=run_forcing_dataset
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12GB
#SBATCH --output=forcing_%A_%a.out
#SBATCH --error=forcing_%A_%a.err
#SBATCH --array=1-275

## This script will run a load of emulator datasets
## for low-res sims with imperfect parameterisations

# Begin execution
module purge

singularity exec --nv \
	    --overlay /scratch/cp3759/sing/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /home/cp3759/Projects/pyqg_explorer/scripts/gen_forcing.py --save_to /scratch/cp3759/pyqg_data/sims/2_step_forcing/2_step_CNN.nc --increment 2 --run_number $SLURM_ARRAY_TASK_ID --parameterization CNN"
