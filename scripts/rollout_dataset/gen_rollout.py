import argparse
import os
import pyqg_explorer.simulations.gen_parameterized_sims as genparam
import pyqg_explorer.simulations.gen_forcing_sims as genforcing
import pyqg_explorer.parameterizations.parameterizations as parameterizations
import pyqg_explorer.util.misc as misc

parser = argparse.ArgumentParser()
parser.add_argument('--save_to', type=str)
parser.add_argument('--increment', type=int, default=0)
parser.add_argument('--rollout', type=int, default=1)
parser.add_argument('--run_number', type=int, default=0)
parser.add_argument('--parameterization', type=str)
args, extra = parser.parse_known_args()

print(args)

if args.parameterization=="CNN":
    ## Default pure-offline CNN
    model_cnn=misc.load_model("/scratch/cp3759/pyqg_data/models/cnn_theta_ONLY_forcing1_both_epoch200.pt")
    parameterization=parameterizations.Parameterization(model_cnn,cache_forcing=True)
elif args.parameterization=="ZB":
    parameterization=parameterizations.ZannaBolton2020Q(cache_forcing=True)
elif args.parameterization=="BScat":
    parameterization=parameterizations.BackscatterBiharmonic(cache_forcing=True)
elif args.parameterization=="None":
    ## If none, run a high res simulation with "perfect" subgrid forcing
    parameterization=None

print(parameterization)

## Add run number to save file name
save_file=list(args.save_to)
save_file.insert(-3,"_"+args.parameterization+"_")
save_file.insert(-3,str(args.run_number))
save_file="".join(save_file)

print(save_file)
if parameterization:
    print("running parameterised")
    ds = genparam.generate_parameterized_dataset(increment=args.increment,rollout=args.rollout,parameterization=parameterization)
else:
    print("running highres")
    ds = genforcing.generate_forcing_dataset(increment=args.increment, rollout=args.rollout)
ds.to_netcdf(save_file)
