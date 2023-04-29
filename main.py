import os
import subprocess
import argparse

from src.train_model import InitHiddenStates, full_run
from src.data_loader import IterativeLoader

import warnings
warnings.filterwarnings("ignore")


def clean_data(filename):
    cmd = f"python utils/preprocess_raw.py -d {filename}"
    subprocess.run(cmd, shell=True)
    
    cmd = f"python utils/preprocess_cleaned.py -d clean_{filename}.csv"
    subprocess.run(cmd, shell=True)
    return


def integrate_cleaned_data():
    cmd = "python utils/preprocess_integrate.py"
    subprocess.run(cmd, shell=True)
    return


def _init_pretrained_params():
    IL = IterativeLoader()
    all_states = IL.get_hidden_states()
    all_params = IL.get_assignment_params()
    
    all_states = all_states.iloc[:, :2]
    all_states.to_hdf("data/hidden_states.h5", key='hs', index=False)
    all_params = all_params.iloc[:, :2]
    all_params.to_hdf("data/assignment_params.h5", key='ap', index=False)


def init_hidden(mode="linear_regression"):
    if os.path.exists("data/hidden_states.h5"):
        os.remove("data/hidden_states.h5")
    
    IL = IterativeLoader()
    
    value, covariate = IL.get_init_hidden_data()

    if mode == "linear_regression":
        IH = InitHiddenStates(value=value, covariate=covariate)
        IH.fit()
        IH.validate()
        IH.label_states()

    if mode == "order":
        IH.do_naive_ordering()
        
    return


def run_all():
    _init_pretrained_params()
    full_run()
    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="methodname")
    parser.add_argument("-d", "--data", help="dataname")
    
    args = parser.parse_args()
    kwargs = vars(args)

    dataname = kwargs['data']
    method = kwargs['method']
    
    allowed_methods = ['clean_data', 'integrate_cleaned_data', 'init_hidden', 'run_all']
    assert method in allowed_methods, f"Unknown -m argument, must be one of {allowed_methods}, got {method} instead"
