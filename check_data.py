import pandas as pd
import numpy as np
import os

base = r"C:\isac_project\LiDAR\Scenario8\development_dataset"

df = pd.read_csv(os.path.join(base, "scenario8_dev_train.csv"))
print(df.head())
print("训练集样本数:", len(df))

row = df.iloc[0]
lidar = np.load(os.path.join(base, row["unit1_lidar_1"]))
print("LiDAR矩阵形状:", lidar.shape)
print("前3行:\n", lidar[:3])

pwr = np.loadtxt(os.path.join(base, row["unit1_pwr_1"]))
print("功率向量长度:", len(pwr))
print("功率最大位置:", int(np.argmax(pwr)), " csv里的beam_index:", row["beam_index"])
