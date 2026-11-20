import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

base = r"C:\isac_project\LiDAR\Scenario8\development_dataset"
df = pd.read_csv(os.path.join(base, "scenario8_dev_train.csv"))
row = df.iloc[0]

pwr = np.loadtxt(os.path.join(base, row["unit1_pwr_1"]))
print("功率向量长度:", len(pwr))
print("功率最大位置:", int(np.argmax(pwr)), " csv里的beam_index_1:", row["beam_index_1"])

d = loadmat(os.path.join(base, row["unit1_lidar_1"]))
for k, v in d.items():
    if k.startswith("__"):
        continue
    print("LiDAR变量名:", k, " 形状:", getattr(v, "shape", None))
    if isinstance(v, np.ndarray):
        print("前3行:\n", v[:3])
