#!/bin/bash

#SBATCH --job-name=run_KE_sim
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH --output=KE_sim_%A_%a.out
#SBATCH --error=KE_sim_%A_%a.err
#SBATCH --array=1-50


# Begin execution
module purge

singularity exec --nv \
	    --overlay /scratch/cp3759/sing/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /home/cp3759/Projects/pyqg_explorer/scripts/KE_datasets/gen_lowres_j.py --save_to /scratch/cp3759/pyqg_data/sims/KE_accumulation/dt5_beta100/alpha_15/lrj.nc --alpha 15  --increment 0 --run_number $SLURM_ARRAY_TASK_ID"
