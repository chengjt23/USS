import importlib
import os
import json
import numpy as np
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)


def get_restore_step(path):
    checkpoints = os.listdir(path)
    if os.path.exists(os.path.join(path, "final.ckpt")):
        return "final.ckpt", 0
    elif not os.path.exists(os.path.join(path, "last.ckpt")):
        steps = [int(x.split(".ckpt")[0].split("step=")[1]) for x in checkpoints]
        return checkpoints[np.argmax(steps)], np.max(steps)
    else:
        steps = []
        for x in checkpoints:
            if "last" in x:
                if "-v" not in x:
                    fname = "last.ckpt"
                else:
                    this_version = int(x.split(".ckpt")[0].split("-v")[1])
                    steps.append(this_version)
                    if len(steps) == 0 or this_version > np.max(steps):
                        fname = "last-v%s.ckpt" % this_version
        return fname, 0
