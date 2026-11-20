import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

base = r"C:\isac_project\LiDAR\Scenario8\development_dataset"
df = pd.read_csv(os.path.join(base, "scenario8_dev_train.csv"))
rel = df["unit1_lidar_SCR_1"].iloc[0]
print("SCR文件:", rel)
d = loadmat(os.path.join(base, rel))["data"]
print("形状:", d.shape)
print("前10行:\n", d[:10])
print("后5行:\n", d[-5:])
for j in range(2):
    c = d[:, j]
    print(f"第{j}列: 最小 {c.min():.4f} 最大 {c.max():.4f} 下降次数 {(np.diff(c) < -1e-6).sum()}")
