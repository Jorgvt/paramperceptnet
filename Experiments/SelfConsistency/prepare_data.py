import pandas as pd

mos = []
with open("mos.txt", "rb") as f:
    for line in f.readlines():
        mos.append(float(line.strip()))

std = []
with open("mos_std.txt", "rb") as f:
    for line in f.readlines():
        std.append(float(line.strip()))

df = pd.DataFrame({
    "mos": mos,
    "std": std,
})
print(df.head())
df.to_csv("tid2013_mos_std.csv", index=False)
