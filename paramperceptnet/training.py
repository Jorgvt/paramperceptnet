from functools import partial

import jax
from jax import numpy as jnp
from flax.core import FrozenDict
from flax import struct
from flax.training import train_state
from clu import metrics


@struct.dataclass
class Metrics(metrics.Collection):
    """Collection of metrics to be tracked during training."""

    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics
    state: FrozenDict


def create_train_state(module, key, tx, input_shape):
    """Creates the initial `TrainState`."""
    variables = module.init(key, jnp.ones(input_shape))
    state, params = variables.pop("params")
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        state=state,
        tx=tx,
        metrics=Metrics.empty(),
    )


def pearson_correlation(vec1, vec2):
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()
    vec1_mean = vec1.mean()
    vec2_mean = vec2.mean()
    num = vec1 - vec1_mean
    num *= vec2 - vec2_mean
    num = num.sum()
    denom = jnp.sqrt(jnp.sum((vec1 - vec1_mean) ** 2))
    denom *= jnp.sqrt(jnp.sum((vec2 - vec2_mean) ** 2))
    return num / denom


@partial(jax.jit, static_argnums=2)
def train_step(state, batch, return_grads=False):
    """Train for a single step."""
    img, img_dist, mos = batch

    def loss_fn(params):
        ## Forward pass through the model
        img_pred, updated_state = state.apply_fn(
            {"params": params, **state.state},
            img,
            mutable=list(state.state.keys()),
            train=True,
        )
        img_dist_pred, updated_state = state.apply_fn(
            {"params": params, **state.state},
            img_dist,
            mutable=list(state.state.keys()),
            train=True,
        )

        ## Calculate the distance
        dist = ((img_pred - img_dist_pred) ** 2).sum(axis=(1, 2, 3)) ** (1 / 2)

        ## Calculate pearson correlation
        return pearson_correlation(dist, mos), updated_state

    (loss, updated_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    state = state.apply_gradients(grads=grads)
    metrics_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    state = state.replace(state=updated_state)
    if return_grads:
        return state, grads
    else:
        return state


@jax.jit
def compute_metrics(*, state, batch):
    """Obtaining the metrics for a given batch."""
    img, img_dist, mos = batch

    def loss_fn(params):
        ## Forward pass through the model
        img_pred, updated_state = state.apply_fn(
            {"params": params, **state.state},
            img,
            mutable=list(state.state.keys()),
            train=False,
        )
        img_dist_pred, updated_state = state.apply_fn(
            {"params": params, **state.state},
            img_dist,
            mutable=list(state.state.keys()),
            train=False,
        )

        ## Calculate the distance
        dist = ((img_pred - img_dist_pred) ** 2).sum(axis=(1, 2, 3)) ** (1 / 2)

        ## Calculate pearson correlation
        return pearson_correlation(dist, mos)

    metrics_updates = state.metrics.single_from_model_output(loss=loss_fn(state.params))
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return state
