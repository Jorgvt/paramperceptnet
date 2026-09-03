"""Obtain per-pair distance predictions for Progressive Freezing models.

Evaluates all 8 progressive freezing stages + the 9th Fully Frozen (0 params / model-0)
stage across TID2008, TID2013, and KADIK10K datasets, saves per-pair predictions
as CSVs, and logs them to Weights & Biases.
"""

import os
# Mask GPU from TensorFlow so it doesn't claim VRAM before JAX
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import argparse
import copy
import glob
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy.stats as stats
import orbax.checkpoint
import wandb
import jax
from jax import numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze
import optax

from iqadatasets.datasets import TID2008, TID2013, KADIK10K
from paramperceptnet.models import PerceptNet
from paramperceptnet.configs import param_config as config
from paramperceptnet.training import create_train_state

# Handle CPU/CUDA sharding deserialization across platforms
import orbax.checkpoint.type_handlers as ocp_type_handlers

try:
    cpu_device = jax.devices("cpu")[0]
    device_map = {str(device): device for device in jax.local_devices()}
    device_map[str(cpu_device)] = cpu_device
    device_map["TFRT_CPU_0"] = cpu_device
    if len(jax.local_devices()) > 0 and jax.local_devices()[0].platform == "gpu":
        device_map["cuda:0"] = jax.local_devices()[0]
    ocp_type_handlers._deserialize_sharding_from_json_string.device_map = device_map
except Exception:
    pass


def _deep_restore(target, source):
    """Recursively overwrite target leaves with source values, preserving keys absent in source."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_restore(target[key], value)
        else:
            target[key] = value


def align(raw_val, target_val):
    """Recursively align checkpoint parameter tree to match dummy state structure."""
    if isinstance(target_val, (FrozenDict, dict)):
        res = {}
        for k, v in target_val.items():
            if k == "LinearScaling_0" and k not in raw_val and "B" in raw_val:
                res[k] = (
                    freeze({"B": raw_val["B"]})
                    if isinstance(target_val, FrozenDict)
                    else {"B": raw_val["B"]}
                )
            elif (
                k == "B"
                and k not in raw_val
                and "LinearScaling_0" in raw_val
                and "B" in raw_val["LinearScaling_0"]
            ):
                res[k] = raw_val["LinearScaling_0"]["B"]
            elif isinstance(raw_val, dict) and k in raw_val:
                res[k] = align(raw_val[k], v)
            else:
                res[k] = v
        return freeze(res) if isinstance(target_val, FrozenDict) else res
    elif isinstance(target_val, tuple):
        if raw_val is None:
            if hasattr(target_val, "_fields") and len(target_val._fields) == 0:
                return type(target_val)()
            return None
        if hasattr(target_val, "_fields"):
            elements = [
                align(raw_val[f], getattr(target_val, f))
                for f in target_val._fields
            ]
            return type(target_val)(*elements)
        if isinstance(raw_val, list):
            elements = [
                align(raw_val[i], target_val[i]) for i in range(len(target_val))
            ]
        elif isinstance(raw_val, dict):
            elements = []
            for i in range(len(target_val)):
                if str(i) in raw_val:
                    elements.append(align(raw_val[str(i)], target_val[i]))
                elif i in raw_val:
                    elements.append(align(raw_val[i], target_val[i]))
                else:
                    elements.append(target_val[i])
        else:
            elements = target_val
        return tuple(elements)
    elif hasattr(target_val, "__dataclass_fields__"):
        f_vals = {}
        for f in target_val.__dataclass_fields__:
            if hasattr(raw_val, f):
                f_vals[f] = align(getattr(raw_val, f), getattr(target_val, f))
            elif isinstance(raw_val, dict) and f in raw_val:
                f_vals[f] = align(raw_val[f], getattr(target_val, f))
            else:
                f_vals[f] = getattr(target_val, f)
        return type(target_val)(**f_vals)
    else:
        return raw_val


@jax.jit
def forward_pass(params, state, img, dist):
    """Compute perceptual distance between reference and distorted images."""
    pnet = PerceptNet(config)
    img_pred = pnet.apply({"params": params, **state}, img, train=False)
    dist_pred = pnet.apply({"params": params, **state}, dist, train=False)
    distance = jnp.sqrt(jnp.mean((img_pred - dist_pred) ** 2, axis=(1, 2, 3)))
    return distance


def evaluate_dataset_predictions(state, tf_dataset):
    """Run model inference on a dataset and return Pearson correlation, predicted distances, and MOS."""
    all_dists = []
    all_mos = []

    for batch in tf_dataset.as_numpy_iterator():
        ref, dist, mos = batch
        pred_dist = forward_pass(state.params, state.state, ref, dist)
        all_dists.extend(np.array(pred_dist))
        all_mos.extend(np.array(mos))

    all_dists = np.array(all_dists)
    all_mos = np.array(all_mos)
    corr, _ = stats.pearsonr(all_dists, all_mos)
    return corr, all_dists, all_mos


DEFAULT_PROGRESSIVE_IDS = [
    ("i8kkltwu", "model-best", "FinalModel_TrainAll_GoodInit", 1062),
    ("gx9gpizs", "model-best", "FinalModel_FreezeGDNGamma_GoodInit", 1060),
    ("c9u2vqjz", "model-best", "FinalModel_FreezeJH_GoodInit", 1051),
    ("2aae1qvd", "model-best", "FinalModel_Freeze_GDNColor_GoodInit", 1045),
    ("f8uv6afu", "model-best", "FinalModel_Freeze_CS_GoodInit", 1018),
    ("3r2slksi", "model-best", "FinalModel_Freeze_GDNGaussian_GoodInit", 1009),
    ("k24dfyo8", "model-best", "FinalModel_GDNFinalOnly_GoodInit", 553),
    ("csrhdpbd", "model-best", "FinalModel_OnlyB_GoodInit", 128),
    ("csrhdpbd", "model-0", "FinalModel_FullyFrozen", 0),
]


def main():
    parser = argparse.ArgumentParser(
        description="Obtain per-pair predictions for progressive freezing models and upload to W&B"
    )
    parser.add_argument(
        "--tid08_path",
        type=str,
        default="/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2008/",
        help="Path to TID2008 dataset",
    )
    parser.add_argument(
        "--tid13_path",
        type=str,
        default="/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2013/",
        help="Path to TID2013 dataset",
    )
    parser.add_argument(
        "--kadid_path",
        type=str,
        default="/media/disk/vista/BBDD_video_image/Image_Quality/KADIK10K/",
        help="Path to KADIK10K dataset",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="Jorgvt/PerceptNet_v15",
        help="W&B project name",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="predictions_progressive",
        help="Local directory to save per-pair prediction CSVs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for dataset evaluation (default: 32)",
    )
    parser.add_argument(
        "--upload_to_wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to upload prediction CSV files to W&B runs (default: True)",
    )
    args = parser.parse_args()

    # 1. Load Datasets
    print("Loading TID2008 dataset...")
    dst_tid08 = TID2008(args.tid08_path, exclude_imgs=[25])
    dst_tid08_rdy = (
        dst_tid08.dataset.batch(args.batch_size, drop_remainder=True)
        .prefetch(1)
    )

    print("Loading TID2013 dataset...")
    dst_tid13 = TID2013(args.tid13_path, exclude_imgs=[25])
    dst_tid13_rdy = (
        dst_tid13.dataset.batch(args.batch_size, drop_remainder=True)
        .prefetch(1)
    )

    print("Loading KADIK10K dataset...")
    dst_kad = KADIK10K(args.kadid_path, exclude_identical_pairs=True)
    dst_kad_rdy = (
        dst_kad.dataset.batch(args.batch_size, drop_remainder=True)
        .prefetch(1)
    )

    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    api = wandb.Api()

    results = []

    for run_id, ckpt_name, stage_name, n_params in DEFAULT_PROGRESSIVE_IDS:
        print(f"\n=======================================================")
        print(f"Evaluating: {stage_name} (Run ID: {run_id}, Checkpoint: {ckpt_name}, Params: {n_params})")
        print(f"=======================================================")

        run = api.run(f"{args.project}/{run_id}")
        run_name = run.name

        checkpoint_dir = None
        ckpt_local_tag = f"{run_id}_{ckpt_name.replace('-', '_')}"

        # Download checkpoint from W&B
        files = [f for f in run.files() if f.name.startswith(f"{ckpt_name}/") or f.name.startswith(f"repaired_checkpoints/repaired/{run_id}/{ckpt_name}/")]
        if len(files) > 0:
            checkpoint_dir = f"wandb_checkpoints/{ckpt_local_tag}/{ckpt_name}"
            print(f"Downloading checkpoint {ckpt_name} for run: {run_name} (ID: {run_id}) from W&B...")
            for f in files:
                f.download(root=f"wandb_checkpoints/{ckpt_local_tag}", replace=True)
        else:
            local_runs = glob.glob(f"wandb/run-*-{run_id}/files/{ckpt_name}/checkpoint")
            if len(local_runs) > 0:
                checkpoint_dir = os.path.dirname(local_runs[0])
                print(f"Using local checkpoint for run {run_name} (ID: {run_id}) at: {checkpoint_dir}")

        if not checkpoint_dir:
            print(f"Checkpoint not found for run {run_name} (ID: {run_id}). Skipping...")
            continue

        try:
            # Build model configuration from run config
            model_config = copy.deepcopy(config)
            if hasattr(run, "config") and run.config:
                for k, v in run.config.items():
                    model_config[k] = v

            model = PerceptNet(model_config)
            dummy_state = create_train_state(
                model,
                jax.random.PRNGKey(model_config.SEED),
                optax.adam(model_config.LEARNING_RATE),
                input_shape=(1, 384, 512, 3),
            )

            raw = ckptr.restore(os.path.abspath(checkpoint_dir))
            aligned_state = align(raw, dummy_state)

            # Precalc filter handling
            if hasattr(aligned_state, "state") and "precalc_filter" in aligned_state.state:
                raw_precalc = dict(raw.get("state", {}).get("precalc_filter", {})) if isinstance(raw, dict) else {}
                _, new_precalc = aligned_state.apply_fn(
                    {"params": aligned_state.params, **aligned_state.state},
                    jnp.ones((1, 384, 512, 3)),
                    train=True,
                    mutable=["precalc_filter"],
                )
                if "precalc_filter" in new_precalc:
                    merged_precalc = dict(new_precalc["precalc_filter"])
                    _deep_restore(merged_precalc, raw_precalc)
                    aligned_state = aligned_state.replace(
                        state={
                            **aligned_state.state,
                            "precalc_filter": merged_precalc,
                        }
                    )

            # Re-evaluate TID2008, TID2013, and KADIK10K
            print("  Evaluating TID2008...")
            tid08_corr, tid08_dists, tid08_mos = evaluate_dataset_predictions(
                aligned_state, dst_tid08_rdy
            )
            print(f"  - TID2008 Pearson: {tid08_corr:.5f}")

            print("  Evaluating TID2013...")
            tid13_corr, tid13_dists, tid13_mos = evaluate_dataset_predictions(
                aligned_state, dst_tid13_rdy
            )
            print(f"  - TID2013 Pearson: {tid13_corr:.5f}")

            print("  Evaluating KADIK10K...")
            kad_corr, kad_dists, kad_mos = evaluate_dataset_predictions(
                aligned_state, dst_kad_rdy
            )
            print(f"  - KADIK10K Pearson: {kad_corr:.5f}")

            # Save per-pair predictions to local CSV files
            run_pred_dir = os.path.join(args.predictions_dir, ckpt_local_tag)
            os.makedirs(run_pred_dir, exist_ok=True)

            df_tid08_pred = pd.DataFrame({
                "pair_idx": np.arange(len(tid08_dists)),
                "predicted_distance": tid08_dists,
                "mos": tid08_mos,
            })
            df_tid13_pred = pd.DataFrame({
                "pair_idx": np.arange(len(tid13_dists)),
                "predicted_distance": tid13_dists,
                "mos": tid13_mos,
            })
            df_kad_pred = pd.DataFrame({
                "pair_idx": np.arange(len(kad_dists)),
                "predicted_distance": kad_dists,
                "mos": kad_mos,
            })

            tid08_csv_path = os.path.join(run_pred_dir, "tid2008_predictions.csv")
            tid13_csv_path = os.path.join(run_pred_dir, "tid2013_predictions.csv")
            kad_csv_path = os.path.join(run_pred_dir, "kadik10k_predictions.csv")

            df_tid08_pred.to_csv(tid08_csv_path, index=False)
            df_tid13_pred.to_csv(tid13_csv_path, index=False)
            df_kad_pred.to_csv(kad_csv_path, index=False)

            if args.upload_to_wandb:
                print(f"  Uploading prediction CSVs to W&B for run {run_id} ({ckpt_name})...")
                run.upload_file(tid08_csv_path, root=args.predictions_dir)
                run.upload_file(tid13_csv_path, root=args.predictions_dir)
                run.upload_file(kad_csv_path, root=args.predictions_dir)
                print("  Upload complete.")

            results.append({
                "run_id": run_id,
                "checkpoint": ckpt_name,
                "stage_name": stage_name,
                "trainable_parameters": n_params,
                "tid2008_corr": tid08_corr,
                "tid2013_corr": tid13_corr,
                "kadik10k_corr": kad_corr,
            })

        except Exception as e:
            print(f"Error processing run {run_id} ({ckpt_name}): {e}")
            import traceback
            traceback.print_exc()

    df_results = pd.DataFrame(results)
    out_csv = "Plotting/progressive_freezing_evaluation_extended.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved evaluation summary table to: {out_csv}")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
