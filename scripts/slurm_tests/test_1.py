
import os

print("attempting to run other script")
os.system("source /scratch/work/public/singularity/greene-ib-slurm-bind.sh")
os.system("sbatch /home/cp3759/Testbed/slurm_tests/submit_second.sh")
