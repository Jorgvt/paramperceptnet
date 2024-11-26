import os
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import argparse
from absl import flags
from absl.flags import FLAGS

import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze, FrozenDict
from flax.training import orbax_utils

import optax
import orbax.checkpoint
from ml_collections import ConfigDict, config_flags

from safetensors.flax import load_file
from huggingface_hub import hf_hub_download
from datasets import load_dataset

from paramperceptnet.models import PerceptNet
from paramperceptnet.constraints import *
from paramperceptnet.training import *


## Load pretrained model
model_name = "ppnet-bio-fitted"
# model_name = "ppnet-fully-trained"
config_path = hf_hub_download(repo_id=f"Jorgvt/{model_name}",
                              filename="config.json")
with open(config_path, "r") as f:
    config = ConfigDict(json.load(f))

weights_path = hf_hub_download(repo_id=f"Jorgvt/{model_name}",
                               filename="weights.safetensors")
variables = load_file(weights_path)
variables = flax.traverse_util.unflatten_dict(variables, sep=".")
state_ = variables["state"]
params_ = variables["params"]

## Prepare datasets
dst_train = load_dataset("Jorgvt/TID2008", trust_remote_code=True)
dst_train = dst_train.with_format("jax")
dst_train = dst_train["train"]

dst_val = load_dataset("Jorgvt/TID2013", trust_remote_code=True)
dst_val = dst_val.with_format("jax")
dst_val = dst_val["train"]

## Define a `TrainState`
tx = optax.adam(config.LEARNING_RATE)
state = create_train_state(
    PerceptNet(config), random.PRNGKey(config.SEED), tx, input_shape=(1, 384, 512, 3)
)
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
state = state.replace(params=clip_param(state.params, "A", a_min=0))


def check_trainable(path):
    if "GDNGamma_0" in path:
        if not config.TRAIN_GDNGAMMA:
            return True
    if "Color" in path:
        if not config.TRAIN_JH:
            return True
    if "GDN_0" in path:
        if not config.TRAIN_GDNCOLOR:
            return True
    if "CenterSurroundLogSigmaK_0" in path:
        if not config.TRAIN_CS:
            return True
    if "GDNGaussian_0" in path:
        if not config.TRAIN_GDNGAUSSIAN:
            return True
    if "Gabor" in "".join(path):
        if not config.TRAIN_GABOR:
            return True
    if not config.A_GDNSPATIOFREQORIENT:
        if ("GDNSpatioChromaFreqOrient_0" in path) and ("A" in path):
            return True
    if "GDNSpatioChromaFreqOrient_0" not in path and config.TRAIN_ONLY_LAST_GDN:
        return True
    return False

trainable_tree = freeze(
    flax.traverse_util.path_aware_map(
        lambda path, v: "non_trainable" if check_trainable(path) else "trainable",
        state.params,
    )
)

optimizers = {
    "trainable": optax.adam(config.LEARNING_RATE),
    "non_trainable": optax.set_to_zero(),
}

tx = optax.multi_transform(optimizers, trainable_tree)

state = create_train_state(
    PerceptNet(config), random.PRNGKey(config.SEED), tx, input_shape=(1, 384, 512, 3)
)
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
state = state.replace(params=clip_param(state.params, "A", a_min=0))


param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
trainable_param_count = sum(
    [
        w.size if t == "trainable" else 0
        for w, t in zip(
            jax.tree_util.tree_leaves(state.params),
            jax.tree_util.tree_leaves(trainable_tree),
        )
    ]
)
print(f"Total parameters: {param_count} | Trainable parameters: {trainable_param_count}")

### Attach the loaded params and state
state = state.replace(state=freeze(state_))
state = state.replace(params=freeze(params_))

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)

metrics_history = {
    "train_loss": [],
    "val_loss": [],
}

for epoch in range(config.EPOCHS):
    ## Training
    for batch in dst_train.iter(batch_size=config.BATCH_SIZE): 
        batch = (batch["reference"], batch["distorted"], batch["mos"])
        state, grads = train_step(state, batch, return_grads=True)
        state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
        state = state.replace(params=clip_param(state.params, "A", a_min=0))
        state = state.replace(params=clip_param(state.params, "K", a_min=1 + 1e-5))

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"train_{name}"].append(value)

    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Evaluation
    for batch in dst_val.iter(batch_size=config.BATCH_SIZE): 
        batch = (batch["reference"], batch["distorted"], batch["mos"])
        state = compute_metrics(state=state, batch=batch)

    for name, value in state.metrics.compute().items():
        metrics_history[f"val_{name}"].append(value)
    state = state.replace(metrics=state.metrics.empty())

    ## Checkpointing
    if metrics_history["val_loss"][-1] <= min(metrics_history["val_loss"]):
        orbax_checkpointer.save(
            "model-best",
            state,
            save_args=save_args,
            force=True,
        ) 

    print(
        f'Epoch {epoch} -> [Train] Loss: {metrics_history["train_loss"][-1]} [Val] Loss: {metrics_history["val_loss"][-1]}'
    )

# Save the final model as well in case we want to keep training from it or whatever:

orbax_checkpointer.save(
    "model-final", state, save_args=save_args
)
