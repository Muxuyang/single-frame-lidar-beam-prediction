import os
import numpy as np
import pandas as pd

base = r"C:\isac_project\LiDAR\Scenario8\development_dataset"
df = pd.read_csv(os.path.join(base, "scenario8_dev_train.csv"))
ok = 0
n = 100
for i in range(n):
    row = df.iloc[i]
    pwr = np.loadtxt(os.path.join(base, row["unit1_pwr_1"]))
    if int(np.argmax(pwr)) + 1 == int(row["beam_index_1"]):
        ok += 1
print(f"{n}条里验证通过: {ok}条")
