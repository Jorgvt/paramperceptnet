import tensorflow as tf

tf.config.set_visible_devices([], device_type="GPU")

import os
import glob
import argparse
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
from tqdm.auto import tqdm

# Monkey-patch Orbax's internal device map cache to handle device sharding differences (CPU/GPU)
import orbax.checkpoint.type_handlers as ocp_type_handlers

try:
    cpu_device = jax.devices("cpu")[0]
    device_map = {str(device): device for device in jax.local_devices()}
    device_map[str(cpu_device)] = cpu_device
    device_map["TFRT_CPU_0"] = cpu_device
    device_map[
        "cuda:0"
    ] = jax.local_devices()[
        0
    ]  # Map cuda:0 to first local device (which is gpu:0 on GPU or cpu:0 on CPU)
    ocp_type_handlers._deserialize_sharding_from_json_string.device_map = (
        device_map
    )
except Exception:
    pass

from iqadatasets.datasets import TID2008, TID2013, KADIK10K
from paramperceptnet.training import create_train_state
from paramperceptnet.configs import param_config as config
from paramperceptnet.models_ablation import AblationPerceptNet


# JIT compile the distance calculation for a batch
@jax.jit
def get_batch_distances(state, img, img_dist):
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


def evaluate_global_correlation(state, dataset_rdy):
    all_dists = []
    all_mos = []
    for batch in dataset_rdy.as_numpy_iterator():
        img, img_dist, mos = batch
        dist = get_batch_distances(state, img, img_dist)
        all_dists.append(dist)
        all_mos.append(mos)

    all_dists = jnp.concatenate(all_dists)
    all_mos = jnp.concatenate(all_mos)
    corr, _ = stats.pearsonr(all_dists, all_mos)
    return corr


def _deep_restore(target: dict, source: dict) -> None:
    """Recursively overwrite target leaves with source values, preserving keys absent in source.

    Used to restore checkpoint-stored kernels after a train=True initialisation pass,
    so that ChromaFreqOrientGaussianGamma (missing from old checkpoints) keeps its
    freshly computed kernel while every other precalc_filter leaf reverts to the
    exact values that were stored at training time.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_restore(target[key], value)
        else:
            target[key] = value


def align(raw_val, target_val):
    if isinstance(target_val, FrozenDict) or isinstance(target_val, dict):
        res = {}
        for k, v in target_val.items():
            # Handle root-level 'B' vs nested 'LinearScaling_0/B' mapping
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


def get_name(config_dict):
    params = []
    if config_dict.get("USE_GAMMA", True):
        params.append("1")
    if config_dict.get("PARAM_CS", True):
        params.append("4")
    if config_dict.get("PARAM_DN_CS", True):
        params.append("5")
    if config_dict.get("PARAM_GABOR", True):
        params.append("6")
    if config_dict.get("PARAM_DN_FINAL", True):
        params.append("7")
    if config_dict.get("FINAL_B", True):
        params.append("8")
    return "[" + ",".join(params) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate parametric ablation variants on datasets"
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

    # 2. Retrieve Runs from WandB
    api = wandb.Api()
    sweep_ids = ["14oohbnh", "10ltg55x", "3825yhz4"]
    print("Fetching sweep runs from W&B...")
    sweeps = [
        api.sweep(f"Jorgvt/PerceptNet_v15/{sweep_id}")
        for sweep_id in sweep_ids
    ]
    runs = [sweep.runs for sweep in sweeps]
    runs = [run for runs_ in runs for run in runs_]
    print(f"Total runs fetched: {len(runs)}")

    histories_list = []
    for run in tqdm(runs, desc="Processing run histories"):
        history = run.history()
        # Keep only runs that successfully finished training
        if len(history) < 500:
            continue

        train_loss = history["train_loss"].min()
        val_loss = history["val_loss"].min()
        run_name_sig = get_name(run.config)

        histories_list.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "name": run_name_sig,
                "config": run.config,
                "run": run,
            }
        )

    # Find the best run per configuration name based on val_loss (minimum val_loss)
    df_runs = pd.DataFrame(histories_list)
    df_best = (
        df_runs.sort_values(by="val_loss", ascending=False)
        .drop_duplicates(subset="name", keep="last")
        .sort_values(by="name")
    )
    print(f"Found {len(df_best)} unique ablation configurations.")

    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    results = []

    # 3. Evaluate each unique configuration
    for idx, row in tqdm(
        df_best.iterrows(),
        total=len(df_best),
        desc="Evaluating configurations",
    ):
        run_id = row["run_id"]
        run_name_sig = row["name"]
        run_config = row["config"]
        run_obj = row["run"]

        # Ensure directories exist
        ckpt_dir = f"wandb_checkpoints/{run_id}/model-best"
        os.makedirs(f"wandb_checkpoints/{run_id}", exist_ok=True)

        # Download checkpoint files
        files = [
            f for f in run_obj.files() if f.name.startswith("model-best/")
        ]
        if len(files) > 0:
            print(
                f"\nDownloading checkpoint for config {run_name_sig} (Run: {run_id})..."
            )
            for f in files:
                f.download(root=f"wandb_checkpoints/{run_id}", replace=True)
        else:
            # Check if it's stored locally
            local_runs = glob.glob(
                f"wandb/run-*-{run_id}/files/model-best/checkpoint"
            )
            if len(local_runs) > 0:
                ckpt_dir = os.path.dirname(local_runs[0])
                print(f"\nUsing local checkpoint for Run: {run_id}...")
            else:
                print(
                    f"\nCheckpoint not found for Run: {run_id}. Skipping..."
                )
                continue

        try:
            # Build corresponding config object
            import copy
            model_config = copy.deepcopy(config)
            for k, v in run_config.items():
                model_config[k] = v

            # Create AblationPerceptNet model and dummy state
            model = AblationPerceptNet(model_config)
            dummy_state = create_train_state(
                model,
                jax.random.PRNGKey(model_config.SEED),
                optax.adam(model_config.LEARNING_RATE),
                input_shape=(1, 384, 512, 3),
            )

            # Restore and align checkpoint weights
            raw = ckptr.restore(os.path.abspath(ckpt_dir))
            aligned_state = align(raw, dummy_state)

            # The old checkpoints were saved without the ChromaFreqOrientGaussianGamma
            # precalc_filter kernel.  Run a train=True pass to compute it, then
            # deep-restore the stored kernels for all other layers so we use the
            # exact values from training time (avoids GPU/XLA float drift).
            if "precalc_filter" in aligned_state.state:
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
                        state={**aligned_state.state, "precalc_filter": merged_precalc}
                    )

            # Count total parameters
            param_count = sum(
                x.size for x in jax.tree_util.tree_leaves(aligned_state.params)
            )

            # Re-evaluate TID2008, TID2013, and evaluate KADIK10K
            tid08_corr = evaluate_global_correlation(
                aligned_state, dst_tid08_rdy
            )
            tid13_corr = evaluate_global_correlation(
                aligned_state, dst_tid13_rdy
            )
            kad_corr = evaluate_global_correlation(aligned_state, dst_kad_rdy)

            print(f"Configuration: {run_name_sig}")
            print(
                f"  - Re-evaluated TID2008 Pearson: {tid08_corr:.4f} (Original loss: {row['train_loss']:.4f})"
            )
            print(
                f"  - Re-evaluated TID2013 Pearson: {tid13_corr:.4f} (Original loss: {row['val_loss']:.4f})"
            )
            print(f"  - KADIK10K Pearson: {kad_corr:.4f}")

            results.append(
                {
                    "name": run_name_sig,
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                    "tid2008_corr": tid08_corr,
                    "tid2013_corr": tid13_corr,
                    "kadik10k_corr": kad_corr,
                    "kadik10k_loss": -kad_corr,
                    "params": param_count,
                    "run_id": run_id,
                }
            )

        except Exception as e:
            print(f"Error evaluating config {run_name_sig} (Run: {run_id}): {e}")

    # 4. Save and output results
    if len(results) > 0:
        df_res = pd.DataFrame(results)
        df_res["len"] = df_res.name.apply(lambda x: len(x))
        df_res = df_res.sort_values(by="len")

        # Save extended results to CSV
        output_csv = "Plotting/results_ablation_parametric_table_extended.csv"
        df_res.to_csv(output_csv, index=False)
        print(f"\nExtended table saved to: {output_csv}")

        # Display LaTeX format matching the original notebook
        print("\n=== Extended LaTeX Table ===")
        latex_str = df_res.filter(
            [
                "name",
                "train_loss",
                "val_loss",
                "kadik10k_loss",
                "kadik10k_corr",
                "params",
            ]
        ).to_latex(
            index=False,
            formatters={"name": lambda x: "$" + x + "$"},
            float_format="%.3f",
        )
        print(latex_str)

        # Save LaTeX code to file
        with open("Plotting/results_ablation_parametric_table_extended.tex", "w") as f:
            f.write(latex_str)

    else:
        print("\nNo configurations were successfully evaluated.")


if __name__ == "__main__":
    main()
