#!/bin/bash

## Script to submit multiple slurm job arrays. Here we want to run
## an ensemble of pyqg simulations, including some ML parameterisation
## set in model_string, for a range of different alphas which we loop over.


save_to="/scratch/cp3759/pyqg_data/sims/KE_accumulation/offline_only_020"
model_string="/scratch/cp3759/pyqg_data/models/cnn_theta_ONLY_forcing1_both_epoch200.pt"
coeff=0.2

## Check if directory exists. If not, make it
if [ ! -d $save_to ]
then
     mkdir $save_to
     echo "Submitting jobs - saving sims in $save_to"
else
    echo "Directory $save_to exists. Quitting to avoid overwriting sims"
    exit
fi

## Write a text file documenting the ML model used in this ensemble, and any additional notes
echo "Notes: offline model only" >> $save_to/notes.txt
echo "Model path used in these runs: $model_string" >> $save_to/notes.txt
echo "Coeff: $coeff" >> $save_to/notes.txt

## Loop over alphas. Submit a job array for each alpha
for alpha in 1 5 10 15
do
   eval sbatch submit_array_cpu.sh --save_to $save_to --model_string $model_string --alpha $alpha --coeff $coeff
done
