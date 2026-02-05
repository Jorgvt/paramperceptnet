from glob import glob

import pandas as pd
import scipy.stats as stats

results = {}
for csv in glob("*csv"):
    name = csv.split(".")[0]
    df = pd.read_csv(csv)
    mos = df.pop("mos")
    pearson = df.apply(lambda x: stats.pearsonr(x, mos)[0])
    results[name] = pearson
    # break

results = pd.DataFrame(results).reset_index(names="Metric")
results
results.to_csv("pearson_results.csv", index=False)