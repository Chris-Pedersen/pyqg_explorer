import argparse
import os
import pyqg_explorer.generate_datasets as generate_datasets
import pyqg_explorer.models.parameterizations as parameterizations
import pyqg_explorer.util.misc as misc

parser = argparse.ArgumentParser()
parser.add_argument('--save_to', type=str)
parser.add_argument('--increment', type=int, default=0)
parser.add_argument('--run_number', type=int, default=0)
parser.add_argument('--parameterization', action="store_true")
args, extra = parser.parse_known_args()

# Setup parameters for dataset generation functions
kwargs = dict()
for param in extra:
    key, val = param.split('=')
    try:
        val = float(val)
    except:
        pass
    kwargs[key.replace('--', '')] = val
files = [f.strip() for f in args.save_to.split(',')]
files = [f for f in files if f]

print(kwargs)

if args.parameterization:
    model_cnn=misc.load_model("/scratch/cp3759/pyqg_data/models/cnn_theta_ONLY_forcing1_both_epoch200.pt")
    parameterization=parameterizations.Parameterization(model_cnn,cache_forcing=True)

for save_file in files:
    ## Add run number to save file name
    save_file=list(save_file)
    save_file.insert(-3,str(args.run_number))
    save_file="".join(save_file)
    os.system(f"mkdir -p {os.path.dirname(os.path.realpath(save_file))}")
    if args.parameterization:
        ds = generate_datasets.generate_parameterized_dataset(parameterization=parameterization,increment=args.increment)
    else:
        ds = generate_datasets.generate_forcing_dataset(**kwargs)
    ds.to_netcdf(save_file)
