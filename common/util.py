import numpy as np
from pathlib import Path
import os
import yaml
from types import SimpleNamespace as SN


def wrapper(x, dim):
    out = np.zeros(dim)
    out[int(x)] = 1
    assert sum(out) == 1, "wrapper: sum of out is not 1!"
    return out


def inverse_wrapper(x):
    assert sum(x) == 1, "inverse_wrapper: sum of x is not 1!"
    x = list(x)
    assert isinstance(x, list), "x is not list, it is " + str(type(x))
    return x.index(1)


def make_logpath(game_name, algo):
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / Path('./models') / game_name.replace('-', '_') / algo

    log_dir = base_dir / Path('./models/config_training')/ game_name.replace('-', '_') / algo
    if not log_dir.exists():
        os.makedirs(log_dir)

    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = log_dir / curr_run

    return run_dir, log_dir

def load_config(dir):
    file = open(os.path.join(dir), "r")
    config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict


def get_paras_from_dict(config_dict):
    dummy_dict = config_reformat(config_dict)
    args = SN(**dummy_dict)
    return args

def config_reformat(my_dict):
    dummy_dict = {}
    for k, v in my_dict.items():
        if type(v) is dict:
            for k2, v2 in v.items():
                dummy_dict[k2] = v2
        else:
            dummy_dict[k] = v
    return dummy_dict
