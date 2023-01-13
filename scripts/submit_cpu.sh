#!/bin/bash

#SBATCH --job-name=run_dataset
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --output=slurm_%j.out


# Begin execution
module purge

singularity exec --nv \
	    --overlay /scratch/cp3759/sing/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /home/cp3759/Projects/pyqg_explorer/scripts/gen_single_dataset.py --save_to /scratch/cp3759/pyqg_data/sims/every_snap.nc --sampling_freq 1 --tmax=10"
