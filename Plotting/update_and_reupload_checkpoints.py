"""Repair and re-upload complete, self-contained checkpoints to Weights & Biases runs.

This script:
1. Targets all 59 manuscript runs:
   - 48 Parametric Ablation models (from sweeps + baseline 396atbyq).
   - 8 Progressive Freezing models.
   - 3 KADID-trained models.
2. For each checkpoint ('model-0', 'model-best', 'model-final'):
   - Downloads/restores the raw checkpoint.
   - Instantiates the proper model architecture and state.
   - Computes all complete precalc_filter kernels (including ChromaFreqOrientGaussianGamma).
   - Deep-restores genuine stored parameter values.
   - Saves a clean, fully self-contained PyTree checkpoint.
   - Re-uploads the complete checkpoint files directly to the W&B run.
"""

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], device_type="GPU")
except ImportError:
    pass

import os
import glob
import copy
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import jax
from jax import numpy as jnp
import orbax.checkpoint
from flax.training import orbax_utils
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

from paramperceptnet.training import create_train_state
from paramperceptnet.configs import param_config as config
from paramperceptnet.models import PerceptNet, Baseline, Original
from paramperceptnet.models_ablation import AblationPerceptNet


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


def collect_target_runs(api, project="Jorgvt/PerceptNet_v15"):
    """Collect all 59 target runs to process: Ablations (48), Progressive Freezing (8), and KADID-trained (3)."""
    runs_to_process = []

    # 1. Ablation Runs (48 runs)
    ablation_csv_path = "Plotting/ablation_bootstrap_results.csv"
    if os.path.exists(ablation_csv_path):
        df_abl = pd.read_csv(ablation_csv_path)
        abl_ids = df_abl["run_id"].unique().tolist()
    else:
        # Fallback list of 48 ablation run IDs
        ext_csv = "Plotting/results_ablation_parametric_table_extended_2.csv"
        df_ext = pd.read_csv(ext_csv)
        abl_ids = df_ext["run_id"].unique().tolist()

    print(f"1. Fetching {len(abl_ids)} Ablation runs...")
    for r_id in tqdm(abl_ids, desc="Ablation runs"):
        try:
            r = api.run(f"{project}/{r_id}")
            runs_to_process.append({
                "run_id": r.id,
                "run_name": r.name,
                "type": "ablation",
                "run_obj": r,
            })
        except Exception as e:
            print(f"  Warning: Could not fetch ablation run {r_id}: {e}")

    # 2. Progressive Freezing Runs (8 runs)
    prog_ids = [
        "i8kkltwu", "gx9gpizs", "c9u2vqjz", "2aae1qvd",
        "f8uv6afu", "3r2slksi", "k24dfyo8", "csrhdpbd",
    ]
    print(f"2. Fetching {len(prog_ids)} Progressive Freezing runs...")
    for r_id in tqdm(prog_ids, desc="Progressive Freezing runs"):
        try:
            r = api.run(f"{project}/{r_id}")
            runs_to_process.append({
                "run_id": r.id,
                "run_name": r.name,
                "type": "progressive",
                "run_obj": r,
            })
        except Exception as e:
            print(f"  Warning: Could not fetch progressive run {r_id}: {e}")

    # 3. KADID-Trained Runs (3 runs)
    kadid_ids = ["ncmk3oal", "ibfurp59", "csste11r"]
    print(f"3. Fetching {len(kadid_ids)} KADID-trained runs...")
    for r_id in tqdm(kadid_ids, desc="KADID-trained runs"):
        try:
            r = api.run(f"{project}/{r_id}")
            runs_to_process.append({
                "run_id": r.id,
                "run_name": r.name,
                "type": "kadid_trained",
                "run_obj": r,
            })
        except Exception as e:
            print(f"  Warning: Could not fetch KADID-trained run {r_id}: {e}")

    print(f"\nTotal runs collected: {len(runs_to_process)} (Target: 59)")
    return runs_to_process


def repair_and_upload_run(run_info, ckpt_types, local_scratch_dir="repaired_checkpoints", upload=True):
    run_id = run_info["run_id"]
    run_name = run_info["run_name"]
    run_type = run_info["type"]
    run_obj = run_info["run_obj"]

    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    repaired_count = 0

    for ckpt_name in ckpt_types:
        local_dl_dir = os.path.join(local_scratch_dir, "raw", run_id, ckpt_name)
        repaired_save_dir = os.path.join(local_scratch_dir, "repaired", run_id, ckpt_name)
        os.makedirs(local_dl_dir, exist_ok=True)
        os.makedirs(repaired_save_dir, exist_ok=True)

        # Download checkpoint files
        files = [f for f in run_obj.files() if f.name.startswith(f"{ckpt_name}/")]
        if len(files) > 0:
            for f in files:
                f.download(root=os.path.join(local_scratch_dir, "raw", run_id), replace=True)
            checkpoint_dir = local_dl_dir
        else:
            local_runs = glob.glob(f"wandb/run-*-{run_id}/files/{ckpt_name}/checkpoint")
            if len(local_runs) > 0:
                checkpoint_dir = os.path.dirname(local_runs[0])
            else:
                # Checkpoint not present on W&B or locally
                continue

        try:
            # Build config
            model_config = copy.deepcopy(config)
            if hasattr(run_obj, "config") and run_obj.config:
                for k, v in run_obj.config.items():
                    model_config[k] = v

            # Determine Architecture
            raw = ckptr.restore(os.path.abspath(checkpoint_dir))
            raw_params = raw.get("params", {})

            if run_type == "ablation":
                model = AblationPerceptNet(model_config)
                dummy_state = create_train_state(
                    model, jax.random.PRNGKey(model_config.SEED), optax.adam(model_config.LEARNING_RATE), input_shape=(1, 384, 512, 3)
                )
            elif "CenterSurroundLogSigmaK_0" in raw_params or "GaborLayerGammaHumanLike__0" in raw_params:
                model = PerceptNet(model_config)
                dummy_state = create_train_state(
                    model, jax.random.PRNGKey(model_config.SEED), optax.adam(model_config.LEARNING_RATE), input_shape=(1, 384, 512, 3)
                )
            elif "Conv_1" in raw_params and "GDN_3" in raw_params:
                model = Baseline(model_config)
                dummy_state = train_state.TrainState.create(
                    apply_fn=model.apply,
                    params=model.init(jax.random.PRNGKey(model_config.SEED), jnp.ones(shape=(1, 384, 512, 3)))["params"],
                    tx=optax.adam(model_config.LEARNING_RATE),
                )
            elif "Color" in raw_params and "GDN_0" in raw_params:
                model = Original(model_config)
                dummy_state = create_train_state(
                    model, jax.random.PRNGKey(model_config.SEED), optax.adam(model_config.LEARNING_RATE), input_shape=(1, 384, 512, 3)
                )
            else:
                model = PerceptNet(model_config)
                dummy_state = create_train_state(
                    model, jax.random.PRNGKey(model_config.SEED), optax.adam(model_config.LEARNING_RATE), input_shape=(1, 384, 512, 3)
                )

            # Align weights
            aligned_state = align(raw, dummy_state)

            # Precalc filter generation & genuine kernel deep restore
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
                        state={**aligned_state.state, "precalc_filter": merged_precalc}
                    )

            # Save the clean, complete PyTree checkpoint
            save_args = orbax_utils.save_args_from_target(aligned_state)
            ckptr.save(
                os.path.abspath(repaired_save_dir),
                aligned_state,
                save_args=save_args,
                force=True,
            )

            # Upload repaired files to W&B
            if upload:
                for root, _, fnames in os.walk(repaired_save_dir):
                    for fname in fnames:
                        full_fpath = os.path.join(root, fname)
                        run_obj.upload_file(full_fpath)

            repaired_count += 1
            print(f"  ✓ Repaired & uploaded [{ckpt_name}] for run {run_name} ({run_id})")

        except Exception as e:
            print(f"  ✗ Error repairing [{ckpt_name}] for run {run_name} ({run_id}): {e}")

    return repaired_count


def main():
    parser = argparse.ArgumentParser(description="Repair and re-upload complete checkpoints to W&B")
    parser.add_argument("--project", type=str, default="Jorgvt/PerceptNet_v15", help="W&B project name")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["model-0", "model-best", "model-final"],
        help="Checkpoint types to process (default: model-0, model-best, model-final)",
    )
    parser.add_argument("--scratch_dir", type=str, default="repaired_checkpoints", help="Local directory for processing")
    parser.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to upload repaired files back to W&B (default: True)",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = collect_target_runs(api, project=args.project)

    print(f"\nStarting checkpoint repair for {len(runs)} runs (Checkpoints: {args.checkpoints})...")
    total_repaired = 0

    for run_info in tqdm(runs, desc="Processing runs"):
        n = repair_and_upload_run(run_info, args.checkpoints, local_scratch_dir=args.scratch_dir, upload=args.upload)
        total_repaired += n

    print(f"\n==========================================")
    print(f"Successfully repaired & uploaded {total_repaired} checkpoints across {len(runs)} runs!")
    print(f"==========================================")


if __name__ == "__main__":
    main()
