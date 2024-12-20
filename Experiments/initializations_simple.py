import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import argparse
from absl import flags
from absl.flags import FLAGS

from typing import Any, Callable, Sequence, Union
from tqdm.auto import tqdm
import numpy as np

import tensorflow as tf
import scipy.stats as stats

tf.config.set_visible_devices([], device_type="GPU")

import jax
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
import pandas as pd

import wandb
from iqadatasets.datasets import *
from JaxPlayground.utils.wandb import *
from paramperceptnet.layers import *
from paramperceptnet.constraints import *
from paramperceptnet.training import *
from paramperceptnet.configs import param_config as config
from paramperceptnet.initialization import humanlike_init

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-N", help="Number of seeds to try", type=int)
parser.add_argument("--model", choices=["parametric", "non-parametric"], help="Two the type of model to use in the experiment.")
parser.add_argument('--path', help="Folder to store the resulting csv files", type=str, default=".")
parser.add_argument("--testing", help="Do only one iteration per dataset", action="store_true")
args = parser.parse_args()

if args.path != ".":
    if not os.path.exists(args.path):
        os.makedirs(args.path)

# %%
dst_train = TID2008("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2008/", exclude_imgs=[25])
dst_val = TID2013("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2013/", exclude_imgs=[25])

# %%
img, img_dist, mos = next(iter(dst_train.dataset))
print(img.shape, img_dist.shape, mos.shape)

# %%
img, img_dist, mos = next(iter(dst_val.dataset))
print(img.shape, img_dist.shape, mos.shape)

# %%
dst_train_rdy = dst_train.dataset.shuffle(
    buffer_size=100, reshuffle_each_iteration=True, seed=config.SEED
).batch(config.BATCH_SIZE, drop_remainder=True)
dst_val_rdy = dst_val.dataset.batch(config.BATCH_SIZE, drop_remainder=True)

if args.model == "parametric":
    from paramperceptnet.models import PerceptNet
    ## Evaluate function
    @jax.jit
    def get_dist(state, img, img_dist):
        pred = state.apply_fn({"params": state.params, **state.state}, img, train=False)
        pred_dist = state.apply_fn({"params": state.params, **state.state}, img_dist, train=False)
        return ((pred-pred_dist)**2).mean(axis=(1,2,3))**(1/2)
elif args.model == "non-parametric":
    from paramperceptnet.models import Baseline as PerceptNet
    ## Evaluate function
    @jax.jit
    def get_dist(state, img, img_dist):
        pred = state.apply_fn({"params": state.params}, img, train=False)
        pred_dist = state.apply_fn({"params": state.params}, img_dist, train=False)
        return ((pred-pred_dist)**2).mean(axis=(1,2,3))**(1/2)

def eval_dst(state, dst):
    dists, moses = [], []
    for batch in dst:
        img, img_dist, mos = batch
        dist = get_dist(state, img, img_dist)
        dists.extend(dist)
        moses.extend(mos)
        if args.testing: break
    return stats.pearsonr(dists, moses)[0]

## Loop
N = args.N
seeds = jnp.linspace(0, 10000, num=N, dtype=int)
results = {"seed":[], "pearson":[]}
for seed in tqdm(seeds):
    state = create_train_state(
        PerceptNet(config), random.PRNGKey(seed), optax.adam(3e-4), input_shape=(1, 384, 512, 3)
    )
    state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
    state = state.replace(params=clip_param(state.params, "A", a_min=0))
    pred, _ = state.apply_fn(
        {"params": state.params, **state.state},
        jnp.ones(shape=(1, 384, 512, 3)),
        train=True,
        mutable=list(state.state.keys()),
    )
    state = state.replace(state=_)

    ## Evaluate the model on the dataset
    res = eval_dst(state, dst_train_rdy.as_numpy_iterator())
    results['seed'].append(int(seed))
    results['pearson'].append(res)

df = pd.DataFrame(results)
df.to_csv(f"{args.path}/{args.model}.csv", index=False)
