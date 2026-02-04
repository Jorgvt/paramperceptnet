import os
from tqdm.auto import tqdm

import pandas as pd
import torch

from piq_metrics import get_all_piq_full_reference_metrics
from iqadatasets.datasets import TID2008

## Fetch all the full reference metrics from PIQ
metrics = get_all_piq_full_reference_metrics(reduction="none")

## Prepare the dataset
data_path = "/media/disk/vista/BBDD_video_image/Image_Quality/"
BATCH_SIZE = 64

dataset = TID2008(path=os.path.join(data_path, "TID", "TID2008"))
dst_rdy = dataset.dataset.batch(BATCH_SIZE).prefetch(1)


results = {}
for name, metric in tqdm(metrics.items(), desc="Metrics", leave=True):
    distances = []
    for img, dist, mos in tqdm(dst_rdy.as_numpy_iterator(), total=len(dst_rdy)):
        img, dist = torch.from_numpy(img,).float(), torch.from_numpy(dist).float()
        img = img.permute(0,3,1,2)
        dist = dist.permute(0,3,1,2)

        with torch.inference_mode():
            distance = metric(img, dist)
        distances.extend(distance.numpy())

        # break

    results[name] = distances
    # break

print(results)
results = pd.DataFrame(results)
