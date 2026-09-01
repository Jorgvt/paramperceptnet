import os
import argparse
import copy
import scipy.stats as stats
import numpy as np
import pandas as pd
import jax
from jax import numpy as jnp
import orbax.checkpoint
import optax
import flax
from flax.training import train_state
from flax.core import freeze, FrozenDict
import wandb

# Monkey-patch Orbax's internal device map cache to handle device sharding differences (CPU/GPU)
import orbax.checkpoint.type_handlers as ocp_type_handlers

try:
    cpu_device = jax.devices("cpu")[0]
    device_map = {str(device): device for device in jax.local_devices()}
    device_map[str(cpu_device)] = cpu_device
    device_map["TFRT_CPU_0"] = cpu_device
    device_map["cuda:0"] = jax.local_devices()[0]
    ocp_type_handlers._deserialize_sharding_from_json_string.device_map = (
        device_map
    )
except Exception:
    pass

from iqadatasets.datasets import TID2008, TID2013, KADIK10K
from paramperceptnet.training import create_train_state
from paramperceptnet.configs import param_config as config
from paramperceptnet.models import PerceptNet, Baseline, Original


# JIT compile distance calculation
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


def evaluate_dataset_predictions(state, dataset_rdy):
    all_dists = []
    all_mos = []
    for batch in dataset_rdy.as_numpy_iterator():
        img, img_dist, mos = batch
        dist = get_batch_distances_perceptnet(state, img, img_dist)
        all_dists.append(np.array(dist))
        all_mos.append(np.array(mos))

    all_dists = np.concatenate(all_dists)
    all_mos = np.concatenate(all_mos)
    corr, _ = stats.pearsonr(all_dists, all_mos)
    return float(corr), all_dists, all_mos


def _deep_restore(target: dict, source: dict) -> None:
    """Recursively overwrite target leaves with source values, preserving keys absent in source."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_restore(target[key], value)
        else:
            target[key] = value


def align(raw_val, target_val):
    if isinstance(target_val, FrozenDict) or isinstance(target_val, dict):
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
        return tuple(elements)
    elif hasattr(target_val, "__dataclass_fields__"):
        f_vals = {}
        for f in target_val.__dataclass_fields__:
            if isinstance(raw_val, dict) and f in raw_val:
                f_vals[f] = align(raw_val[f], getattr(target_val, f))
            else:
                f_vals[f] = getattr(target_val, f)
        return type(target_val)(**f_vals)
    else:
        return raw_val


DEFAULT_PROGRESSIVE_IDS = [
    "i8kkltwu",  # FinalModel_TrainAll_GoodInit
    "gx9gpizs",  # FinalModel_FreezeGDNGamma_GoodInit
    "c9u2vqjz",  # FinalModel_FreezeJH_GoodInit
    "2aae1qvd",  # FinalModel_Freeze_GDNColor_GoodInit
    "f8uv6afu",  # FinalModel_Freeze_CS_GoodInit
    "3r2slksi",  # FinalModel_Freeze_GDNGaussian_GoodInit
    "k24dfyo8",  # FinalModel_GDNFinalOnly_GoodInit
    "csrhdpbd",  # FinalModel_OnlyB_GoodInit
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
        "--run_ids",
        nargs="+",
        default=DEFAULT_PROGRESSIVE_IDS,
        help="List of progressive freezing run IDs",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="predictions_progressive",
        help="Local directory to save per-pair prediction CSVs",
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
        dst_tid08.dataset.batch(config.BATCH_SIZE, drop_remainder=True)
        .prefetch(1)
        .cache()
    )

    print("Loading TID2013 dataset...")
    dst_tid13 = TID2013(args.tid13_path, exclude_imgs=[25])
    dst_tid13_rdy = (
        dst_tid13.dataset.batch(config.BATCH_SIZE, drop_remainder=True)
        .prefetch(1)
        .cache()
    )

    print("Loading KADIK10K dataset...")
    dst_kad = KADIK10K(args.kadid_path, exclude_identical_pairs=True)
    dst_kad_rdy = (
        dst_kad.dataset.batch(config.BATCH_SIZE, drop_remainder=True)
        .prefetch(1)
        .cache()
    )

    ckptr = orbax.checkpoint.PyTreeCheckpointer()

    # 2. Query W&B API for runs
    api = wandb.Api()
    print(f"Fetching {len(args.run_ids)} progressive freezing runs from W&B project '{args.project}'...")
    runs = [api.run(f"{args.project}/{run_id}") for run_id in args.run_ids]

    results = []

    for run in runs:
        run_id = run.id
        run_name = run.name

        ckpt_name = "model-best"
        checkpoint_dir = None

        # Download checkpoint from W&B
        files = [f for f in run.files() if f.name.startswith(f"{ckpt_name}/")]
        if len(files) > 0:
            checkpoint_dir = f"wandb_checkpoints/{run_id}/{ckpt_name}"
            print(f"\nDownloading checkpoint {ckpt_name} for run: {run_name} (ID: {run_id}) from W&B...")
            for f in files:
                f.download(root=f"wandb_checkpoints/{run_id}", replace=True)
        else:
            local_runs = glob.glob(f"wandb/run-*-{run_id}/files/{ckpt_name}/checkpoint")
            if len(local_runs) > 0:
                checkpoint_dir = os.path.dirname(local_runs[0])
                print(f"\nUsing local checkpoint for run {run_name} (ID: {run_id}) at: {checkpoint_dir}")

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
                ckpt_precalc = dict(aligned_state.state.get("precalc_filter", {}))
                _, new_precalc = aligned_state.apply_fn(
                    {"params": aligned_state.params, **aligned_state.state},
                    jnp.ones((1, 384, 512, 3)),
                    train=True,
                    mutable=["precalc_filter"],
                )
                if "precalc_filter" in new_precalc:
                    merged_precalc = dict(new_precalc["precalc_filter"])
                    _deep_restore(merged_precalc, ckpt_precalc)
                    aligned_state = aligned_state.replace(
                        state={
                            **aligned_state.state,
                            "precalc_filter": merged_precalc,
                        }
                    )

            # Evaluate datasets
            tid08_corr, tid08_dists, tid08_mos = evaluate_dataset_predictions(
                aligned_state, dst_tid08_rdy
            )
            tid13_corr, tid13_dists, tid13_mos = evaluate_dataset_predictions(
                aligned_state, dst_tid13_rdy
            )
            kad_corr, kad_dists, kad_mos = evaluate_dataset_predictions(
                aligned_state, dst_kad_rdy
            )

            print(f"[{run_name}]")
            print(f"  - TID2008 Pearson: {tid08_corr:.5f}")
            print(f"  - TID2013 Pearson: {tid13_corr:.5f}")
            print(f"  - KADIK10K Pearson: {kad_corr:.5f}")

            # Save per-pair predictions to local CSV files
            run_pred_dir = os.path.join(args.predictions_dir, run_id)
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

            # Upload CSVs to W&B
            if args.upload_to_wandb:
                try:
                    print(f"  - Uploading prediction CSVs to W&B for run {run_id}...")
                    run.upload_file(tid08_csv_path)
                    run.upload_file(tid13_csv_path)
                    run.upload_file(kad_csv_path)
                    print("  - Upload complete.")
                except Exception as e:
                    print(f"  - Warning: Failed to upload CSVs to W&B for run {run_id}: {e}")

            results.append({
                "Run ID": run_id,
                "Run Name": run_name,
                "Trainable Params": run.summary.get("trainable_parameters", np.nan),
                "TID2008 Correlation": f"{tid08_corr:.5f}",
                "TID2013 Correlation": f"{tid13_corr:.5f}",
                "KADID10K Correlation": f"{kad_corr:.5f}",
            })

        except Exception as e:
            print(f"Error processing run {run_id}: {e}")

    # 3. Print and Save summary
    if len(results) > 0:
        df = pd.DataFrame(results)
        print("\n=== Summary Table ===")
        print(df.to_string(index=False))
        df.to_csv("Plotting/progressive_freezing_correlations.csv", index=False)


if __name__ == "__main__":
    main()
