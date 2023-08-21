#!/bin/bash

#SBATCH --job-name=test_submit.py
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --output=slurm_%j.out

source ~/.bashrc
# Begin execution
module purge

singularity exec --nv \
	    --overlay /scratch/cp3759/sing/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /home/cp3759/Testbed/slurm_tests/test_2.py"