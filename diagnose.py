import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from collections import Counter

base = r"C:\isac_project\LiDAR\Scenario8\development_dataset"
df = pd.read_csv(os.path.join(base, "scenario8_dev_train.csv"))

# 1. 角度网格跨帧是否一致
angs = []
for rel in df["unit1_lidar_1"].iloc[:50]:
    d = loadmat(os.path.join(base, rel))["data"]
    angs.append(np.sort(d[:, 0]))
angs = np.stack(angs)
print("50帧角度网格最大偏差:", np.abs(angs - angs[0]).max())
print("角度范围:", angs[0].min(), angs[0].max())

# 2. 标签分布
cnt = Counter(df["beam_index_1"])
print("最常见的10个波束:", cnt.most_common(10))
print("不同波束数:", len(cnt))

# 3. 相邻行是否时间相邻
def dist_vec(rel):
    d = loadmat(os.path.join(base, rel))["data"]
    return d[np.argsort(d[:, 0]), 1]
v0 = dist_vec(df["unit1_lidar_1"].iloc[0])
v1 = dist_vec(df["unit1_lidar_1"].iloc[1])
vr = dist_vec(df["unit1_lidar_1"].iloc[100])
print("相邻行距离向量差:", np.abs(v0 - v1).mean(), " 随机行差:", np.abs(v0 - vr).mean())
