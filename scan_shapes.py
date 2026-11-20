import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from collections import Counter

base = r"C:\isac_project\LiDAR\Scenario8\development_dataset"

for name in ["scenario8_dev_train.csv", "scenario8_dev_val.csv", "scenario8_dev_test.csv"]:
    df = pd.read_csv(os.path.join(base, name))
    print(name, "行数:", len(df))

df = pd.read_csv(os.path.join(base, "scenario8_dev_train.csv"))
print("train列名:", list(df.columns))
shapes = Counter()
for n, rel in enumerate(df["unit1_lidar_1"]):
    d = loadmat(os.path.join(base, rel))["data"]
    shapes[d.shape] += 1
    if n % 500 == 0:
        print("已扫:", n)
print("训练集LiDAR形状统计:", dict(shapes))

if "unit1_lidar_SCR_1" in df.columns:
    s = loadmat(os.path.join(base, df["unit1_lidar_SCR_1"].iloc[0]))
    for k, v in s.items():
        if not k.startswith("__"):
            print("SCR变量名:", k, " 形状:", getattr(v, "shape", None))
else:
    print("train里没有SCR列")
