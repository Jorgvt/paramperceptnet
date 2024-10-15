import jax
from jax import random, numpy as jnp

from models import PerceptNet
from configs import param_config

print(param_config)
model = PerceptNet(param_config)

variables = model.init(random.PRNGKey(42), jnp.ones((1, 32, 32, 3)))
shapes = jax.tree_util.tree_map(lambda x: x.shape, variables)
print(shapes)
