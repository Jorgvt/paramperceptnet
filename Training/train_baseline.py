import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import argparse
from absl import flags
from absl.flags import FLAGS

from typing import Any, Callable, Sequence, Union
import numpy as np

import tensorflow as tf

tf.config.set_visible_devices([], device_type="GPU")

import jax
# Monkey-patch Orbax's internal device map cache to resolve CPU-to-GPU device mapping differences (TFRT_CPU_0)
import orbax.checkpoint.type_handlers as ocp_type_handlers
try:
    cpu_device = jax.devices('cpu')[0]
    device_map = {str(device): device for device in jax.local_devices()}
    device_map[str(cpu_device)] = cpu_device
    device_map["TFRT_CPU_0"] = cpu_device
    ocp_type_handlers._deserialize_sharding_from_json_string.device_map = device_map
except Exception:
    pass

from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze, FrozenDict
from flax import linen as nn
from flax import struct
from flax.training import train_state
from flax.training import orbax_utils

import optax
import orbax.checkpoint

from clu import metrics
from ml_collections import ConfigDict, config_flags

import wandb
from iqadatasets.datasets import *
from JaxPlayground.utils.wandb import *
from paramperceptnet.models import Baseline as PerceptNet
from paramperceptnet.constraints import *
from paramperceptnet.training import *
from paramperceptnet.configs import param_config as config
from paramperceptnet.initialization import humanlike_init

# _CONFIG = config_flags.DEFINE_config_file("config")
# flags.FLAGS(sys.argv)
# config = _CONFIG.value
print(config)
# %%
# dst_train = TID2008(
#     "/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2008/", exclude_imgs=[25]
# )
# dst_train = KADIK10K("/lustre/ific.uv.es/ml/uv075/Databases/IQA/KADIK10K/")
# dst_val = TID2013(
#     "/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2013/", exclude_imgs=[25]
# )
# dst_train = TID2008("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2013/", exclude_imgs=[25])
dst_train = KADIK10K("/media/disk/vista/BBDD_video_image/Image_Quality/KADIK10K", exclude_identical_pairs=True)
dst_val = TID2013("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2013/", exclude_imgs=[25])
# dst_train = TID2008("/media/databases/IQA/TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/media/databases/IQA/TID/TID2013/", exclude_imgs=[25])

# %%
img, img_dist, mos = next(iter(dst_train.dataset))
img.shape, img_dist.shape, mos.shape

# %%
img, img_dist, mos = next(iter(dst_val.dataset))
img.shape, img_dist.shape, mos.shape

# %%
parser = argparse.ArgumentParser(description="Train PerceptNet Baseline")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from.")
parser.add_argument("--resume_wandb", action="store_true", help="Resume the wandb run associated with the checkpoint.")
args, unknown = parser.parse_known_args()

def parse_wandb_id_from_path(path):
    if not path:
        return None
    parts = path.split('/')
    for part in parts:
        if part.startswith("run-") and "-" in part:
            return part.split("-")[-1]
    return None

wandb_id = None
resume_mode = None
if args.resume and args.resume_wandb:
    wandb_id = parse_wandb_id_from_path(args.resume)
    if wandb_id:
        resume_mode = "must"
        print(f"Resuming wandb run with ID: {wandb_id}")

wandb.init(
    project="PerceptNet_v15",
    name="Baseline_ParamConfig-KADID10K",
    job_type="training-kadid10k",
    config=config,
    mode="online",
    id=wandb_id,
    resume=resume_mode,
)
config = config
config

# %%
dst_train_rdy = dst_train.dataset.shuffle(
    buffer_size=100, reshuffle_each_iteration=True, seed=config.SEED
).batch(config.BATCH_SIZE, drop_remainder=True)
dst_val_rdy = dst_val.dataset.batch(config.BATCH_SIZE, drop_remainder=True)

if hasattr(config, "LEARNING_RATE"):
    tx = optax.adam(config.LEARNING_RATE)
else:
    tx = optax.adam(config.PEAK_LR)
state = create_train_state(
    PerceptNet(config), random.PRNGKey(config.SEED), tx, input_shape=(1, 384, 512, 3)
)
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
state = state.replace(params=clip_param(state.params, "A", a_min=0))

# %%
state.params.keys()

# %%
trainable_tree = freeze(
    flax.traverse_util.path_aware_map(
        lambda path, v: "trainable",
        state.params,
    )
)
print(trainable_tree)

# %%
if hasattr(config, "LEARNING_RATE"):
    tx = optax.adam(learning_rate=config.LEARNING_RATE)
else:
    steps_per_epoch = len(dst_train_rdy)
    epochs = 500
    schedule_lr = optax.warmup_cosine_decay_schedule(
        init_value=config.INITIAL_LR,
        peak_value=config.PEAK_LR,
        end_value=config.END_LR,
        warmup_steps=steps_per_epoch * config.WARMUP_EPOCHS,
        decay_steps=steps_per_epoch * config.EPOCHS,
    )
    tx = optax.adam(learning_rate=schedule_lr)

optimizers = {
    "trainable": tx,
    "non_trainable": optax.set_to_zero(),
}

# %%
tx = optax.multi_transform(optimizers, trainable_tree)

# %%
state = create_train_state(
    PerceptNet(config), random.PRNGKey(config.SEED), tx, input_shape=(1, 384, 512, 3)
)
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))

# %%
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
print(param_count, trainable_param_count)

wandb.run.summary["total_parameters"] = param_count
wandb.run.summary["trainable_parameters"] = trainable_param_count

# %%
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)

# %%
if args.resume:
    # Restore the raw state first
    raw_restored = orbax_checkpointer.restore(args.resume)
    
    # Define target-compatible restore args from raw restored structure
    def make_restore_args(x):
        if hasattr(x, 'sharding') or isinstance(x, jax.Array) or (hasattr(x, 'shape') and len(x.shape) > 0):
            return orbax.checkpoint.ArrayRestoreArgs(
                restore_type=jax.Array,
                sharding=jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
            )
        return orbax.checkpoint.RestoreArgs()
    
    restore_args = jax.tree_util.tree_map(make_restore_args, raw_restored)
    raw_restored = orbax_checkpointer.restore(args.resume, restore_args=restore_args)
    
    # Align raw structures back to the state container types
    def align_structures(raw, target):
        if isinstance(target, FrozenDict):
            return freeze({k: align_structures(raw[k], v) for k, v in target.items()})
        elif isinstance(target, dict):
            return {k: align_structures(raw[k], v) for k, v in target.items()}
        elif isinstance(target, tuple):
            if raw is None:
                if hasattr(target, '_fields') and len(target._fields) == 0:
                    return type(target)()
                return None
            
            # If target is a namedtuple
            if hasattr(target, '_fields'):
                elements = [align_structures(raw[f], getattr(target, f)) for f in target._fields]
                return type(target)(*elements)
                
            # If target is a plain tuple
            if isinstance(raw, list):
                elements = [align_structures(raw[i], target[i]) for i in range(len(target))]
            elif isinstance(raw, dict):
                elements = [align_structures(raw[str(i)], target[i]) for i in range(len(target))]
            else:
                raise TypeError(f"Expected list/dict for tuple, got {type(raw)}")
            return tuple(elements)
        elif isinstance(target, list):
            return [align_structures(raw[i], target[i]) for i in range(len(target))]
        elif hasattr(target, '__dataclass_fields__'):
            field_values = {}
            for f in target.__dataclass_fields__:
                if isinstance(raw, dict) and f in raw:
                    field_values[f] = align_structures(raw[f], getattr(target, f))
                else:
                    field_values[f] = getattr(target, f)
            return type(target)(**field_values)
        else:
            return raw
            
    state = align_structures(raw_restored, state)
    step = int(state.step)
    steps_per_epoch = len(dst_train_rdy)
    start_epoch = step // steps_per_epoch
    print(f"Resumed from checkpoint {args.resume} at step {step} (epoch {start_epoch})")
else:
    orbax_checkpointer.save(
        os.path.join(wandb.run.dir, "model-0"), state, save_args=save_args, force=True
    )  # force=True means allow overwritting.
    wandb.save(os.path.join(wandb.run.dir, "model-0/**/*"), base_path=wandb.run.dir, policy="live")
    step = 0
    start_epoch = 0

# %%
metrics_history = {
    "train_loss": [],
    "val_loss": [],
}
if args.resume:
    # Evaluate the restored model to populate the initial best val loss
    for batch in dst_val_rdy.as_numpy_iterator():
        state = compute_metrics(state=state, batch=batch)
    initial_val_loss = float(state.metrics.compute()["loss"])
    state = state.replace(metrics=state.metrics.empty())
    metrics_history["val_loss"].append(initial_val_loss)
    metrics_history["train_loss"].append(float('inf'))
    print(f"Restored model validation loss: {initial_val_loss}")

# %%
batch = next(iter(dst_train_rdy.as_numpy_iterator()))

# %%
from functools import partial


# %%
@jax.jit
def forward(state, inputs):
    return state.apply_fn({"params": state.params, **state.state}, inputs, train=False)


# %%
@jax.jit
def forward_intermediates(state, inputs):
    return state.apply_fn(
        {"params": state.params, **state.state},
        inputs,
        train=False,
        capture_intermediates=True,
    )


# %%
# %%time
outputs = forward(state, batch[0])
outputs.shape

# %%
# %%time
s1, grads = train_step(state, batch, return_grads=True)

# %%
# jax.config.update("jax_debug_nans", True)


# %%
def filter_extra(extra):
    def filter_intermediates(path, x):
        path = "/".join(path)
        if "Gabor" in path:
            return (x[0][0],)
        else:
            return x

    extra = unfreeze(extra)
    extra["intermediates"] = flax.traverse_util.path_aware_map(
        filter_intermediates, extra["intermediates"]
    )
    return freeze(extra)


# %%
# %%time
for epoch in range(start_epoch, config.EPOCHS):
    ## Training
    for batch in dst_train_rdy.as_numpy_iterator():
        state, grads = train_step(state, batch, return_grads=True)
        state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
        state = state.replace(params=clip_param(state.params, "A", a_min=0))
        state = state.replace(params=clip_param(state.params, "K", a_min=1 + 1e-5))
        wandb.log(
            {f"{k}_grad": wandb.Histogram(v) for k, v in flatten_params(grads).items()},
            commit=False,
        )
        step += 1
        # state = compute_metrics(state=state, batch=batch)
        # break

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"train_{name}"].append(value)

    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Evaluation
    for batch in dst_val_rdy.as_numpy_iterator():
        state = compute_metrics(state=state, batch=batch)
        # break
    for name, value in state.metrics.compute().items():
        metrics_history[f"val_{name}"].append(value)
    state = state.replace(metrics=state.metrics.empty())

    ## Obtain activations of last validation batch
    _, extra = forward_intermediates(state, batch[0])
    extra = filter_extra(extra)  ## Needed because the Gabor layer has multiple outputs

    ## Checkpointing
    if metrics_history["val_loss"][-1] <= min(metrics_history["val_loss"]):
        orbax_checkpointer.save(
            os.path.join(wandb.run.dir, "model-best"),
            state,
            save_args=save_args,
            force=True,
        )  # force=True means allow overwritting.
        wandb.save(
            os.path.join(wandb.run.dir, "model-best/**/*"),
            base_path=wandb.run.dir,
            policy="live",
        )
    # orbax_checkpointer.save(os.path.join(wandb.run.dir, f"model-{epoch+1}"), state, save_args=save_args, force=False) # force=True means allow overwritting.

    wandb.log(
        {f"{k}": wandb.Histogram(v) for k, v in flatten_params(state.params).items()},
        commit=False,
    )
    wandb.log(
        {
            f"{k}": wandb.Histogram(v)
            for k, v in flatten_params(extra["intermediates"]).items()
        },
        commit=False,
    )
    if hasattr(config, "LEARNING_RATE"):
        wandb.log(
            {
                "epoch": epoch + 1,
                "learning_rate": config.LEARNING_RATE,
                **{name: values[-1] for name, values in metrics_history.items()},
            }
        )
    else:
        wandb.log(
            {
                "epoch": epoch + 1,
                "learning_rate": schedule_lr(step),
                **{name: values[-1] for name, values in metrics_history.items()},
            }
        )
    print(
        f'Epoch {epoch} -> [Train] Loss: {metrics_history["train_loss"][-1]} [Val] Loss: {metrics_history["val_loss"][-1]}'
    )
    # break


# %% [markdown]
# Save the final model as well in case we want to keep training from it or whatever:

# %%
orbax_checkpointer.save(
    os.path.join(wandb.run.dir, "model-final"), state, save_args=save_args
)

# %%
wandb.save(
    os.path.join(wandb.run.dir, "model-final/**/*"),
    base_path=wandb.run.dir,
    policy="live",
)
wandb.finish()
