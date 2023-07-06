#!/bin/bash

#SBATCH --job-name=submit_other_jobs_lol
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --output=submit_%j.out


save_to="/scratch/cp3759/pyqg_data/sims/KE_accumulation/test_dev/test_1"
model_string="/scratch/cp3759/pyqg_data/models/cnn_theta_ONLY_forcing1_both_epoch200.pt"

## Check if directory exists. If not, make it
if [ ! -d $save_to ]
then
     mkdir $save_to
     echo "Submitting jobs - saving sims in $save_to"
else
    echo "Directory $save_to exists. Quitting to avoid overwriting sims"
    exit
fi

for alpha in 1 5 10 15
do
   sbatch submit_array_cpu.sh --save_to $save_to --model_string $model_string --alpha $alpha
done