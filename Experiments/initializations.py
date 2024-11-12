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
from paramperceptnet.models import PerceptNet
from paramperceptnet.layers import *
from paramperceptnet.constraints import *
from paramperceptnet.training import *
from paramperceptnet.configs import param_config as config
from paramperceptnet.initialization import humanlike_init

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-N", help="Number of seeds to try", type=int)
parser.add_argument('--path', help="Folder to store the resulting csv files", type=str, default=".")
parser.add_argument("--testing", help="Do only one iteration per dataset", action="store_true")
args = parser.parse_args()

if args.path != ".":
    if not os.path.exists(args.path):
        os.makedirs(args.path)

# %%
# dst_train = TID2008(
#     "/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2008/", exclude_imgs=[25]
# )
# dst_train = KADIK10K("/lustre/ific.uv.es/ml/uv075/Databases/IQA/KADIK10K/")
# dst_val = TID2013(
#     "/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2013/", exclude_imgs=[25]
# )
dst_train = TID2008("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2008/", exclude_imgs=[25])
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
dst_train_rdy = dst_train.dataset.shuffle(
    buffer_size=100, reshuffle_each_iteration=True, seed=config.SEED
).batch(config.BATCH_SIZE, drop_remainder=True)
dst_val_rdy = dst_val.dataset.batch(config.BATCH_SIZE, drop_remainder=True)

if hasattr(config, "LEARNING_RATE"):
    tx = optax.adam(config.LEARNING_RATE)
else:
    tx = optax.adam(config.PEAK_LR)


## Defining the configs
configs = [dict(config).copy() for _ in range(4)]
configs[0]["USE_GAMMA"] = True
configs[0]["PARAM_CS"] = False
configs[0]["PARAM_DN_CS"] = False
configs[0]["PARAM_GABOR"] = False
### 
configs[1]["USE_GAMMA"] = True
configs[1]["PARAM_CS"] = True
configs[1]["PARAM_DN_CS"] = False
configs[1]["PARAM_GABOR"] = False
### 
configs[2]["USE_GAMMA"] = True
configs[2]["PARAM_CS"] = True
configs[2]["PARAM_DN_CS"] = True
configs[2]["PARAM_GABOR"] = False
### 
configs[3]["USE_GAMMA"] = True
configs[3]["PARAM_CS"] = True
configs[3]["PARAM_DN_CS"] = True
configs[3]["PARAM_GABOR"] = True
###
configs = [ConfigDict(c) for c in configs]
print("Configs created!")

class PerceptNet(nn.Module):
    """IQA model inspired by the visual system."""
    
    config: Any

    @nn.compact
    def __call__(
        self,
        inputs,  # Assuming fs = 128 (cpd)
        **kwargs,
    ):
        if self.config.USE_GAMMA:
            outputs = GDNGamma()(inputs)
        else:
            outputs = GDN(kernel_size=(1, 1), apply_independently=True)(inputs)

        outputs = nn.Conv(features=3, kernel_size=(1, 1), use_bias=False, name="Color")(
            outputs
        )
        outputs = nn.max_pool(outputs, window_shape=(2, 2), strides=(2, 2))

        outputs = GDN(kernel_size=(1, 1), apply_independently=True)(outputs)

        outputs = pad_same_from_kernel_size(
            outputs, kernel_size=self.config.CS_KERNEL_SIZE, mode="symmetric"
        )
        if self.config.PARAM_CS:
            outputs = CenterSurroundLogSigmaK(
                features=3,
                kernel_size=self.config.CS_KERNEL_SIZE,
                fs=21,
                use_bias=False,
                padding="VALID",
            )(outputs, **kwargs)
        else:
            outputs = nn.Conv(features=3, kernel_size=(self.config.CS_KERNEL_SIZE, self.config.CS_KERNEL_SIZE), use_bias=False, padding="VALID")(outputs)

        outputs = nn.max_pool(outputs, window_shape=(2, 2), strides=(2, 2))

        if self.config.PARAM_DN_CS:
            outputs = GDNGaussian(
                kernel_size=self.config.GDNGAUSSIAN_KERNEL_SIZE,
                apply_independently=True,
                fs=32,
                padding="symmetric",
                normalize_prob=self.config.NORMALIZE_PROB,
                normalize_energy=self.config.NORMALIZE_ENERGY,
            )(outputs, **kwargs)
        else:
            outputs = GDN(kernel_size=(self.config.GDNGAUSSIAN_KERNEL_SIZE,self.config.GDNGAUSSIAN_KERNEL_SIZE), apply_independently=True)(outputs)

        outputs = pad_same_from_kernel_size(
            outputs, kernel_size=self.config.GABOR_KERNEL_SIZE, mode="symmetric"
        )

        if self.config.PARAM_GABOR:
            outputs, fmean, theta_mean = GaborLayerGammaHumanLike_(
                n_scales=[4, 2, 2],
                n_orientations=[8, 8, 8],
                kernel_size=self.config.GABOR_KERNEL_SIZE,
                fs=32,
                xmean=self.config.GABOR_KERNEL_SIZE / 32 / 2,
                ymean=self.config.GABOR_KERNEL_SIZE / 32 / 2,
                strides=1,
                padding="VALID",
                normalize_prob=self.config.NORMALIZE_PROB,
                normalize_energy=self.config.NORMALIZE_ENERGY,
                zero_mean=self.config.ZERO_MEAN,
                use_bias=self.config.USE_BIAS,
                train_A=self.config.A_GABOR,
            )(outputs, return_freq=True, return_theta=True, **kwargs)
            outputs = GDNSpatioChromaFreqOrient(
                kernel_size=21,
                strides=1,
                padding="symmetric",
                fs=32,
                apply_independently=False,
                normalize_prob=self.config.NORMALIZE_PROB,
                normalize_energy=self.config.NORMALIZE_ENERGY,
            )(outputs, fmean=fmean, theta_mean=theta_mean, **kwargs)
        else:
            outputs = nn.Conv(features=128, kernel_size=(self.config.GABOR_KERNEL_SIZE, self.config.GABOR_KERNEL_SIZE), use_bias=False, padding="VALID")(outputs)
            outputs = GDN(kernel_size=(self.config.GABOR_KERNEL_SIZE, self.config.GABOR_KERNEL_SIZE), apply_independently=False)(outputs)

        return outputs

## Evaluate function
@jax.jit
def get_dist(state, img, img_dist):
    pred = state.apply_fn({"params": state.params, **state.state}, img, train=False)
    pred_dist = state.apply_fn({"params": state.params, **state.state}, img_dist, train=False)
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
seeds = random.randint(key=random.PRNGKey(42), shape=(N,), minval=0, maxval=1000)
for config, name in tqdm(zip(configs, ["GAMMA", "CS", "DN_CS", "GABOR"])):
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
    df.to_csv(f"{args.path}/{name}.csv", index=False)
