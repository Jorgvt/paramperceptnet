import os

import jax
import orbax.checkpoint
import wandb
from ml_collections import ConfigDict
from datasets import load_dataset
import scipy.stats as stats

from paramperceptnet.models import PerceptNet

## Load the data
dst_train = load_dataset("Jorgvt/TID2008", trust_remote_code=True)
dst_train = dst_train.with_format("jax")
dst_train = dst_train["train"]

dst_val = load_dataset("Jorgvt/TID2013", trust_remote_code=True)
dst_val = dst_val.with_format("jax")
dst_val = dst_val["train"]

## Set up the required ids
ids = [#"afr86ups", # No Param
       "i8kkltwu", # TrainAll_GoodInit
       "csrhdpbd", # OnlyB_GoodInit
    ]


api = wandb.Api()

# runs = api.runs("Jorgvt/PerceptNet_v15")
runs = [api.run(f"Jorgvt/PerceptNet_v15/{id}") for id in ids]

@jax.jit
def compute_distance(params, state, img, dist):
    img_pred = model.apply({"params": params, **state}, img, train=False)
    dist_pred = model.apply({"params": params, **state}, dist, train=False)
    dist = ((img_pred - dist_pred)**2).mean(axis=(1,2,3))**(1/2)
    return dist

for run in runs:
    ## 1. Fetch config
    config = run.config
    if "_fields" in config.keys():
        config = config["_fields"]
    config = ConfigDict(config)
    ## 2. Instantiate model
    model = PerceptNet(config)
    ## 3. Download initial weights
    for file in run.files():
        file.download(root="./models/", replace=True)
    ## 4. Generate the filters
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data = orbax_checkpointer.restore(os.path.join("./models/","model-0"))
    params = data["params"]
    state = data["state"]
    ## 5. Evaluate
    distances, moses = [], []
    for batch in dst_train.iter(batch_size=config.BATCH_SIZE):
        batch = (batch["reference"], batch["distorted"], batch["mos"])
        distance = compute_distance(params, state, batch[0], batch[1])
        distances.extend(distance)
        moses.extend(batch[2])
        # break
    print(f"id: {run.id} | Corr: {stats.pearsonr(distances, moses)[0]}")
    # break
