import argparse
import pyqg_explorer.generate_datasets as generate_datasets
import pyqg_explorer.util.misc as misc
import pyqg_explorer.parameterizations.parameterizations as parameterizations
import os

parser = argparse.ArgumentParser()
parser.add_argument('--save_to', type=str)
parser.add_argument('--model_string', type=str)       ## Path to trained CNN parameterisation
parser.add_argument('--run_number', type=int, default=1)
parser.add_argument('--alpha', type=int, default=1)   ## Coefficient modifying ssd filter
parser.add_argument('--coeff', type=float, default=1) ## Coefficient to attenuate ML model
args, extra = parser.parse_known_args()

# Parse arguments into kwarg dict
kwargs = dict()
YEAR = 24*60*60*360.
kwargs["nx"]=64
kwargs["filterfac"]=23.6*args.alpha
kwargs["tmax"]=tmax=20*YEAR

## Load specificed CNN parameterisation
model_cnn=misc.load_model(args.model_string)
kwargs["parameterization"]=parameterizations.Parameterization(model_cnn,coeff=args.coeff)

## Add run number to save file name
save_to=list(args.save_to)
save_to.append("/alpha_"+str(args.alpha)+"_")
save_to.append("run_"+str(args.run_number)+".nc")
save_to="".join(save_to)
print(kwargs)
print(save_to)

## Run sim
ds = generate_datasets.generate_dataset(**kwargs)
ds.to_netcdf(save_to)
