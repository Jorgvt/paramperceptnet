import tensorflow as tf
tf.config.set_visible_devices([], device_type="GPU")

import jax
from jax import numpy as jnp
# Monkey-patch Orbax's internal device map cache to handle device sharding differences (CPU/GPU)
import orbax.checkpoint.type_handlers as ocp_type_handlers
try:
    cpu_device = jax.devices('cpu')[0]
    device_map = {str(device): device for device in jax.local_devices()}
    device_map[str(cpu_device)] = cpu_device
    device_map["TFRT_CPU_0"] = cpu_device
    device_map["cuda:0"] = jax.local_devices()[0] # Map cuda:0 to first local device (which is gpu:0 on GPU or cpu:0 on CPU)
    ocp_type_handlers._deserialize_sharding_from_json_string.device_map = device_map
except Exception:
    pass

import os
import glob
import argparse
import scipy.stats as stats
import orbax.checkpoint
import optax
import flax
from flax.training import train_state
import wandb
import pandas as pd
from flax.core import freeze, FrozenDict
from iqadatasets.datasets import TID2008, TID2013, KADIK10K
from paramperceptnet.training import create_train_state
from paramperceptnet.configs import param_config as config
from paramperceptnet.models import PerceptNet, Baseline

# JIT compile the distance calculation for a batch (PerceptNet)
@jax.jit
def get_batch_distances_perceptnet(state, img, img_dist):
    img_pred = state.apply_fn(
        {"params": state.params, **state.state},
        img,
        train=False,
    )
    img_dist_pred = state.apply_fn(
        {"params": state.params, **state.state},
        img_dist,
        train=False,
    )
    dist = ((img_pred - img_dist_pred) ** 2).sum(axis=(1, 2, 3)) ** (1 / 2)
    return dist

# JIT compile the distance calculation for a batch (Baseline model - which does not have state)
@jax.jit
def get_batch_distances_baseline(state, img, img_dist):
    img_pred = state.apply_fn(
        {"params": state.params},
        img,
        train=False,
    )
    img_dist_pred = state.apply_fn(
        {"params": state.params},
        img_dist,
        train=False,
    )
    dist = ((img_pred - img_dist_pred) ** 2).sum(axis=(1, 2, 3)) ** (1 / 2)
    return dist

def evaluate_global_correlation(state, dataset_rdy, is_baseline):
    all_dists = []
    all_mos = []
    for batch in dataset_rdy.as_numpy_iterator():
        img, img_dist, mos = batch
        if is_baseline:
            dist = get_batch_distances_baseline(state, img, img_dist)
        else:
            dist = get_batch_distances_perceptnet(state, img, img_dist)
        all_dists.append(dist)
        all_mos.append(mos)
    
    all_dists = jnp.concatenate(all_dists)
    all_mos = jnp.concatenate(all_mos)
    corr, _ = stats.pearsonr(all_dists, all_mos)
    return corr

def align(raw_val, target_val):
    if isinstance(target_val, FrozenDict):
        res = {}
        for k, v in target_val.items():
            if isinstance(raw_val, dict) and k in raw_val:
                res[k] = align(raw_val[k], v)
            else:
                res[k] = v
        return freeze(res)
    elif isinstance(target_val, dict):
        res = {}
        for k, v in target_val.items():
            if isinstance(raw_val, dict) and k in raw_val:
                res[k] = align(raw_val[k], v)
            else:
                res[k] = v
        return res
    elif isinstance(target_val, tuple):
        if raw_val is None:
            if hasattr(target_val, '_fields') and len(target_val._fields) == 0:
                return type(target_val)()
            return None
        if hasattr(target_val, '_fields'):
            elements = [align(raw_val[f], getattr(target_val, f)) for f in target_val._fields]
            return type(target_val)(*elements)
        if isinstance(raw_val, list):
            elements = [align(raw_val[i], target_val[i]) for i in range(len(target_val))]
        elif isinstance(raw_val, dict):
            elements = []
            for i in range(len(target_val)):
                if str(i) in raw_val:
                    elements.append(align(raw_val[str(i)], target_val[i]))
                elif i in raw_val:
                    elements.append(align(raw_val[i], target_val[i]))
                else:
                    elements.append(target_val[i])
        return tuple(elements)
    elif hasattr(target_val, '__dataclass_fields__'):
        f_vals = {}
        for f in target_val.__dataclass_fields__:
            if isinstance(raw_val, dict) and f in raw_val:
                f_vals[f] = align(raw_val[f], getattr(target_val, f))
            else:
                f_vals[f] = getattr(target_val, f)
        return type(target_val)(**f_vals)
    else:
        return raw_val

def main():
    parser = argparse.ArgumentParser(description="Evaluate KADID-trained models globally")
    parser.add_argument("--tid08_path", type=str, default="/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2008/", help="Path to TID2008 dataset")
    parser.add_argument("--tid13_path", type=str, default="/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2013/", help="Path to TID2013 dataset")
    parser.add_argument("--kadid_path", type=str, default="/media/disk/vista/BBDD_video_image/Image_Quality/KADIK10K", help="Path to KADID10K dataset")
    args = parser.parse_args()

    # 1. Load Datasets
    print("Loading TID2008 dataset...")
    dst_tid08 = TID2008(args.tid08_path, exclude_imgs=[25])
    dst_tid08_rdy = dst_tid08.dataset.batch(config.BATCH_SIZE, drop_remainder=True).prefetch(1)

    print("Loading TID2013 dataset...")
    dst_tid13 = TID2013(args.tid13_path, exclude_imgs=[25])
    dst_tid13_rdy = dst_tid13.dataset.batch(config.BATCH_SIZE, drop_remainder=True).prefetch(1)

    print("Loading KADIK10K dataset...")
    dst_kad = KADIK10K(args.kadid_path, exclude_identical_pairs=True)
    dst_kad_rdy = dst_kad.dataset.batch(config.BATCH_SIZE, drop_remainder=True).prefetch(1)

    ckptr = orbax.checkpoint.PyTreeCheckpointer()

    # 2. Query W&B API for runs in the project
    api = wandb.Api()
    print("Fetching runs from W&B project 'jorgvt/PerceptNet_v15'...")
    runs = api.runs("jorgvt/PerceptNet_v15", order="-created_at")
    
    results = []
    
    for run in runs:
        run_id = run.id
        run_name = run.name
        job_type = run.job_type or ""
        
        # Check if trained on KADID
        dataset_name = run.config.get("dataset", "")
        is_kadid = "kadid" in str(dataset_name).lower() or "kadik" in str(dataset_name).lower() or "kadid" in str(job_type).lower() or "kadid" in run_name.lower()
        
        if not is_kadid:
            # Double check config in case it is saved in config object
            cfg_obj = run.config.get("config", {})
            if isinstance(cfg_obj, dict):
                if "kadid" in str(cfg_obj.get("dataset", "")).lower() or "kadik" in str(cfg_obj.get("dataset", "")).lower():
                    is_kadid = True
                    
        if not is_kadid:
            continue
            
        # Determine checkpoint path and download/locate it
        checkpoint_dir = None
        
        # First, try finding model-best files on W&B
        files = [f for f in run.files() if f.name.startswith("model-best/")]
        if len(files) > 0:
            checkpoint_dir = f"wandb_checkpoints/{run_id}/model-best"
            print(f"\nDownloading checkpoint for KADID run: {run_name} (ID: {run_id}) from W&B...")
            for f in files:
                f.download(root=f"wandb_checkpoints/{run_id}", replace=True)
        else:
            # Fall back to local search in case files were not synced to W&B
            local_runs = glob.glob(f"wandb/run-*-{run_id}/files/model-best")
            if len(local_runs) > 0:
                checkpoint_dir = local_runs[0]
                print(f"\nNo checkpoint on W&B for run {run_name} (ID: {run_id}), using local copy at: {checkpoint_dir}")
                
        if not checkpoint_dir:
            # If no checkpoint found, skip
            continue
            
        try:
            # Check if this is the convolutional Baseline or PerceptNet
            is_baseline = "baseline" in run_name.lower() or "baseline" in job_type.lower()
            
            # Build corresponding dummy state to restore checkpoint
            if is_baseline:
                model = Baseline(config)
                dummy_state = train_state.TrainState.create(
                    apply_fn=model.apply,
                    params=model.init(jax.random.PRNGKey(config.SEED), jnp.ones(shape=(1, 384, 512, 3)))["params"],
                    tx=optax.adam(config.LEARNING_RATE)
                )
            else:
                model = PerceptNet(config)
                dummy_state = create_train_state(
                    model, jax.random.PRNGKey(config.SEED), optax.adam(config.LEARNING_RATE), input_shape=(1, 384, 512, 3)
                )
                
            # Restore and align checkpoint
            raw = ckptr.restore(os.path.abspath(checkpoint_dir))
            aligned_state = align(raw, dummy_state)
            
            # Evaluate TID2013 and KADID10K globally
            tid08_corr = evaluate_global_correlation(aligned_state, dst_tid08_rdy, is_baseline)
            tid13_corr = evaluate_global_correlation(aligned_state, dst_tid13_rdy, is_baseline)
            kad_corr = evaluate_global_correlation(aligned_state, dst_kad_rdy, is_baseline)
            
            print(f"-> Global TID2008 Correlation: {tid08_corr:.5f}")
            print(f"-> Global TID2013 Correlation: {tid13_corr:.5f}")
            print(f"-> Global KADID10K Correlation: {kad_corr:.5f}")
            
            results.append({
                "Run ID": run_id,
                "Run Name": run_name,
                "Job Type": job_type,
                "Model Type": "Baseline Conv" if is_baseline else "PerceptNet",
                "TID2008 Correlation": f"{tid08_corr:.5f}",
                "TID2013 Correlation": f"{tid13_corr:.5f}",
                "KADID10K Correlation": f"{kad_corr:.5f}",
                "Checkpoint Path": checkpoint_dir
            })
            
        except Exception as e:
            print(f"Error processing run {run_id}: {e}")
            
    # 3. Print and Save results
    if len(results) > 0:
        df = pd.DataFrame(results)
        markdown_table = df.to_markdown(index=False)
        
        output_text = f"# Local KADID-Trained Models Global Correlation\n\n{markdown_table}\n"
        
        # Save to local workspace
        with open("local_kadid_correlations.md", "w") as f:
            f.write(output_text)
            
        df.to_csv("local_kadid_correlations.csv", index=False)
            
        print("\n=== Summary Table ===")
        print(markdown_table)
        print("\nResults successfully saved to:")
        print("  - Training/local_kadid_correlations.md")
        print("  - Training/local_kadid_correlations.csv")
    else:
        print("\nNo local KADID-trained runs with checkpoints found.")

if __name__ == "__main__":
    main()
