import os
import argparse
from tqdm.auto import tqdm

import pandas as pd
import torch

import tensorflow as tf
tf.config.set_visible_devices([], device_type="GPU")
from iqadatasets.datasets import *

from piq_metrics import get_all_piq_full_reference_metrics


parser = argparse.ArgumentParser(description="Benchmark PIQ metrics on TID2008")
parser.add_argument("--data_path", type=str, default="/media/disk/vista/BBDD_video_image/Image_Quality/", help="Path to the dataset root")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the dataset")
parser.add_argument("--name", type=str, default=None, help="Name for the results CSV file (defaults to data_path basename)")
parser.add_argument("--dst", type=str, default=None, help="Dataset that we want to calculate")
args = parser.parse_args()

## Check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Fetch all the full reference metrics from PIQ
metrics = get_all_piq_full_reference_metrics(reduction="none")

## Prepare the dataset
data_path = args.data_path
BATCH_SIZE = args.batch_size
file_name = args.name if args.name is not None else os.path.basename(data_path.rstrip(os.sep))
dst_name = args.dst

print(f"Data path: {data_path}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Name: {file_name}")

dataset = eval(f"{dst_name}(path='{data_path}')")
dst_rdy = dataset.dataset.batch(BATCH_SIZE).prefetch(1)


is_mos_stored = False
results = {}
for name, metric in tqdm(metrics.items(), desc="Metrics", leave=True):
    metric = metric.to(device)
    distances = []

    if not is_mos_stored: moses = []

    for img, dist, mos in tqdm(dst_rdy.as_numpy_iterator(), total=len(dst_rdy)):
        img, dist = torch.from_numpy(img,).float(), torch.from_numpy(dist).float()
        img = img.permute(0,3,1,2)
        dist = dist.permute(0,3,1,2)
        img, dist = img.to(device), dist.to(device)

        with torch.inference_mode():
            distance = metric(img, dist)
        distances.extend(distance.cpu().numpy())
        if not is_mos_stored: moses.extend(mos)

        # break

    results[name] = distances
    if not is_mos_stored:
        results["mos"] = moses
        is_mos_stored = True
    # break

results = pd.DataFrame(results)
results.to_csv(f"{file_name}.csv", index=False)
