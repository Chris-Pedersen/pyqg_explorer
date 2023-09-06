#!/bin/bash

#SBATCH --job-name=run_KE_sim
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH --output=/scratch/cp3759/pyqg_data/sims/KE_accumulation/slurm_files/KE_sim_%A_%a.out
#SBATCH --error=/scratch/cp3759/pyqg_data/sims/KE_accumulation/slurm_files/KE_sim_%A_%a.err
#SBATCH --array=1-10

source ~/.bashrc
module purge

## Parse variable arguments from command line
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --save_to) save_to="$2"; shift ;;
        --model_string) model_string="$2" ;;
        --alpha) alpha="$2" ;;
        --coeff) coeff="$2" ;;
    esac
    shift
done

## Execute python script using variable arguments
singularity exec --nv \
	    --overlay /scratch/cp3759/sing/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /home/cp3759/Projects/pyqg_explorer/scripts/KE_datasets/run_sim.py --save_to $save_to --model_string $model_string --alpha $alpha --run_number $SLURM_ARRAY_TASK_ID --coeff $coeff"
