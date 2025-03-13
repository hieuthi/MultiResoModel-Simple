import json
import re
import random
import os
from typing import List, Tuple, Optional

import numpy as np
import torch

from torch import Tensor
from torch.nn import functional as F, Module


def reproducibility(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn_deterministic = True
    cudnn_benchmark = False
    print("cudnn_deterministic set to False")
    print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return

def read_json(path: str, object_hook=None):
    with open(path, 'r') as f:
        return json.load(f, object_hook=object_hook)

def save_checkpoint(ckpt_path, modules, meta):
    save_modules = {}
    for name in modules:
        save_modules[name] = modules[name].state_dict()
    states = {"modules": save_modules, "meta": meta}
    torch.save(states, ckpt_path )

def load_checkpoint(ckpt_path, modules):
    states = torch.load(ckpt_path)
    for name in modules:
        modules[name].load_state_dict(states["modules"][name])
    return modules, states["meta"]
