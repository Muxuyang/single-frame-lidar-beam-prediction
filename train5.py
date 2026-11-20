import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

BASE = r"C:\isac_project\LiDAR\Scenario8\development_dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

# SCR文件列顺序（探针实测）：第0列是距离，第1列是角度；(0,0)行是剔除点，保留为0
SCR_DIST_COL = 0

def power_soft_target(raw, T=2.0):
    """Ma式功率软目标：功率向量经温度缩放后归一化（若原始值为dB负数，先转线性）"""
    p = raw.astype(np.float64)
    if p.min() < 0:
        p = 10 ** (p / 10.0)
    p = np.maximum(p, 1e-12)
    q = p ** (1.0 / T)
    return (q / q.sum()).astype(np.float32)

def load_split(csv_name):
    df = pd.read_csv(os.path.join(BASE, csv_name))
    xs, ys, ps = [], [], []
    for _, row in df.iterrows():
        d = loadmat(os.path.join(BASE, row["unit1_lidar_SCR_1"]))["data"]
        xs.append(d[:, SCR_DIST_COL].astype(np.float32))
        ys.append(int(row["beam_index_1"]) - 1)
        raw = np.loadtxt(os.path.join(BASE, row["unit1_pwr_1"])).astype(np.float64)
        if len(ps) == 0:
            print(csv_name, "首条功率原始值前5:", raw[:5], "范围:", raw.min(), raw.max())
        ps.append(power_soft_target(raw))
    return np.stack(xs), np.array(ys), np.stack(ps)

class BeamDataset(Dataset):
    def __init__(self, x, y, p):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.p = torch.from_numpy(p)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i], self.p[i]

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

@torch.no_grad()
def topk_acc(model, loader, ks=(1, 3)):
    model.eval()
    correct = {k: 0 for k in ks}
    total = 0
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        top = torch.topk(logits, max(ks), dim=1).indices
        for k in ks:
            correct[k] += (top[:, :k] == y.unsqueeze(1)).any(dim=1).sum().item()
        total += len(y)
    return {k: correct[k] / total for k in ks}

def main():
    print("设备:", DEVICE)
    print("加载数据...")
    x_train, y_train, p_train = load_split("scenario8_dev_train.csv")
    x_val, y_val, p_val = load_split("scenario8_dev_val.csv")
    x_test, y_test, p_test = load_split("scenario8_dev_test.csv")
    print("train/val/test:", x_train.shape, x_val.shape, x_test.shape)

    train_loader = DataLoader(BeamDataset(x_train, y_train, p_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(BeamDataset(x_val, y_val, p_val), batch_size=256)
    test_loader = DataLoader(BeamDataset(x_test, y_test, p_test), batch_size=256)

    model = BeamNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_top1 = 0.0
    for epoch in range(30):
        model.train()
        tot_loss, n = 0.0, 0
        for x, y, p in train_loader:
            x, y, p = x.to(DEVICE), y.to(DEVICE), p.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = -(p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(y)
            n += len(y)
        acc = topk_acc(model, val_loader)
        print(f"epoch {epoch+1:2d}  train loss {tot_loss/n:.4f}  val Top-1 {acc[1]*100:.2f}%  Top-3 {acc[3]*100:.2f}%")
        if acc[1] > best_top1:
            best_top1 = acc[1]
            torch.save(model.state_dict(), "best_model_pow.pt")

    model.load_state_dict(torch.load("best_model_pow.pt", map_location=DEVICE))
    test_acc = topk_acc(model, test_loader)
    print(f"测试集（最佳模型） Top-1 {test_acc[1]*100:.2f}%  Top-3 {test_acc[3]*100:.2f}%")
    print("baseline对照：train2 Top-1 58.35% / Top-3 90.62%")

if __name__ == "__main__":
    main()
