import json

import jax
from jax import random, numpy as jnp
import flax
from huggingface_hub import hf_hub_download
from ml_collections import ConfigDict
from safetensors.flax import load_file

from paramperceptnet.configs import param_config


def load_param_pretrained(model_name="ppnet-bio-fitted"):
    from paramperceptnet.models import PerceptNet as PerceptNet

    config_path = hf_hub_download(repo_id=f"Jorgvt/{model_name}",
                                filename="config.json")
    with open(config_path, "r") as f:
        config = ConfigDict(json.load(f))

    weights_path = hf_hub_download(repo_id=f"Jorgvt/{model_name}",
                                filename="weights.safetensors")
    variables = load_file(weights_path)
    variables = flax.traverse_util.unflatten_dict(variables, sep=".")

    model = PerceptNet(config)

    return model, variables

def load_baseline_pretrained(model_name="ppnet-baseline"):
    from paramperceptnet.models import Baseline as PerceptNet

    config_path = hf_hub_download(repo_id=f"Jorgvt/{model_name}",
                                filename="config.json")
    with open(config_path, "r") as f:
        config = ConfigDict(json.load(f))

    weights_path = hf_hub_download(repo_id=f"Jorgvt/{model_name}",
                                filename="weights.safetensors")
    variables = load_file(weights_path)
    variables = flax.traverse_util.unflatten_dict(variables, sep=".")

    model = PerceptNet(config)

    return model, variables
