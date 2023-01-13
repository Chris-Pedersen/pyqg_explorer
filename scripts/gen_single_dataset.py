import argparse
import pyqg_explorer.generate_datasets as generate_datasets
import os

parser = argparse.ArgumentParser()
parser.add_argument('--save_to', type=str)
parser.add_argument('--run_number', type=int, default=1)
parser.add_argument('--sampling_freq', type=int, default=1000)
## Will produce two outputs, first for listed arguments above, the second
## containing any additional arguments (such as pyqg arguments)
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

for save_file in files:
    ## Add run number to save file name
    save_file=list(save_file)
    save_file.insert(-3,str(args.run_number))
    save_file="".join(save_file)
    os.system(f"mkdir -p {os.path.dirname(os.path.realpath(save_file))}")
    ds = generate_datasets.generate_dataset(sampling_freq=args.sampling_freq,**kwargs)
    print("here")
    ds.to_netcdf(save_file)