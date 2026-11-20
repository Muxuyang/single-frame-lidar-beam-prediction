import os, re, random
import numpy as np
import pandas as pd
from scipy.io import loadmat

base = r"C:\isac_project\LiDAR\Scenario8\development_dataset"
df = pd.read_csv(os.path.join(base, "scenario8_dev_train.csv"))
df["num"] = df["unit1_lidar_1"].apply(lambda r: int(re.search(r"(\d+)", os.path.basename(r)).group(1)))
print("编号范围:", df["num"].min(), df["num"].max(), "唯一数:", df["num"].nunique(), "总行数:", len(df))
s = df.sort_values("num").reset_index(drop=True)

def dist_vec(rel):
    d = loadmat(os.path.join(base, rel))["data"]
    return d[np.argsort(d[:, 0]), 1]

diffs_adj, diffs_rand = [], []
for i in range(30):
    a = dist_vec(s["unit1_lidar_1"].iloc[i])
    b = dist_vec(s["unit1_lidar_1"].iloc[i + 1])
    diffs_adj.append(np.abs(a - b).mean())
    c = dist_vec(s["unit1_lidar_1"].iloc[random.randrange(len(s))])
    diffs_rand.append(np.abs(a - c).mean())
print("按编号相邻差均值:", np.mean(diffs_adj), " 随机差均值:", np.mean(diffs_rand))
