Because this isn't immediately obvious here and I keep forgetting when I come back to it: the purpose of these scripts is to run an ensemble of simulations, parameterised with a provided CNN (we provide path to model), for different values of alpha, which set the diffusivity of the ssd filter in pyqg. We run 50 simulations by default for each alpha, and save them to a provided path. The script call hierarchy is as follows:

1. We call `sbatch submit_ensembles.sh`.
2. This then calls `sbatch submit_array_cpu.sh`, which will submit an array of 50 jobs for each alpha. The alpha, model, and simulation save directories are all passed as arguments to this script
3. Each job array will call `run_sim.py`, which will load the ML parameterisation and run each individual simulation. Job array means these will be done in parallel on seperate nodes.
4. `run_sim.py` then uses our methods from `generate_datasets`.


