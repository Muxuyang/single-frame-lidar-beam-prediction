import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat
from collections import Counter

BASE = r"C:\isac_project\LiDAR\Scenario8\development_dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCR_DIST_COL = 0

class BeamNet(nn.Module):
    def __init__(self, d_in=216, d_e=64, n_beam=64):
        super().__init__()
        self.embed = nn.Linear(d_in, d_e)
        self.gru = nn.GRU(d_e, d_e, batch_first=True)
        self.cls = nn.Linear(d_e, n_beam)
    def forward(self, x):
        e = self.embed(x).unsqueeze(1)
        out, _ = self.gru(e)
        return self.cls(out[:, -1, :])

df = pd.read_csv(os.path.join(BASE, "scenario8_dev_test.csv"))
xs, ys = [], []
for _, row in df.iterrows():
    d = loadmat(os.path.join(BASE, row["unit1_lidar_SCR_1"]))["data"]
    xs.append(d[:, SCR_DIST_COL].astype(np.float32))
    ys.append(int(row["beam_index_1"]) - 1)
x = torch.from_numpy(np.stack(xs)).to(DEVICE)
y = np.array(ys)

model = BeamNet().to(DEVICE)
model.load_state_dict(torch.load("best_model_scr.pt", map_location=DEVICE))
model.eval()
with torch.no_grad():
    logits = model(x)
    top3 = torch.topk(logits, 3, dim=1).indices.cpu().numpy()
pred = top3[:, 0]

top1_ok = pred == y
print("test样本数:", len(y), " Top-1正确:", top1_ok.sum(), " 错误:", (~top1_ok).sum())
err_beams = Counter(y[~top1_ok])
print("错误最多的真值波束Top10:", err_beams.most_common(10))
top3_ok = np.array([y[i] in top3[i] for i in range(len(y))])
print("Top-1错但Top-3对:", ((~top1_ok) & top3_ok).sum(), " Top-3也错:", ((~top1_ok) & (~top3_ok)).sum())
for b in sorted(set(y)):
    m = y == b
    if m.sum() >= 10:
        print(f"波束{b+1}: 样本{m.sum()}  Top-1准确率 {top1_ok[m].mean()*100:.1f}%")
