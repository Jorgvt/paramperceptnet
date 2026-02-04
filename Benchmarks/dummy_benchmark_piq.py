from tqdm.auto import tqdm
import torch

from piq_metrics import get_all_piq_full_reference_metrics
from benchmark import dummy_dataset
# from iqadatasets.datasets import TID2008

## Fetch all the full reference metrics from PIQ
metrics = get_all_piq_full_reference_metrics(reduction="none")

## Prepare the dataset
# dataset = TID2008(path="")

results = {}
for name, metric in tqdm(metrics.items(), desc="Metrics", leave=True):
    distances = []
    for img, dist, mos in tqdm(dummy_dataset(img_size=(256,256,3)), desc="Dataset", leave=False):
        img, dist = torch.from_numpy(img,).float(), torch.from_numpy(dist).float()
        img = img.permute(0,3,1,2)
        dist = dist.permute(0,3,1,2)

        with torch.inference_mode():
            distance = metric(img, dist)
        distances.extend(distance.numpy())

        break

    results[name] = distances
    # break

print(results)
