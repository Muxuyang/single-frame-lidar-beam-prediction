import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

BASE = r"C:\isac_project\LiDAR\Scenario8\development_dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

def load_split(csv_name):
    df = pd.read_csv(os.path.join(BASE, csv_name))
    xs, ys = [], []
    for _, row in df.iterrows():
        d = loadmat(os.path.join(BASE, row["unit1_lidar_1"]))["data"]
        d = d[np.argsort(d[:, 0])]  # 按角度排序，保证第i个元素对应第i个角度
        xs.append(d[:, 1].astype(np.float32))  # 只取距离列
        ys.append(int(row["beam_index_1"]) - 1)  # MATLAB 1-based -> Python 0-based
    return np.stack(xs), np.array(ys)

class BeamDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

class BeamNet(nn.Module):
    def __init__(self, d_in=460, d_e=64, n_beam=64):
        super().__init__()
        self.embed = nn.Linear(d_in, d_e)
        self.gru = nn.GRU(d_e, d_e, batch_first=True)
        self.cls = nn.Linear(d_e, n_beam)
    def forward(self, x):
        e = self.embed(x).unsqueeze(1)      # (B, 1, 64)
        out, _ = self.gru(e)                # (B, 1, 64)
        return self.cls(out[:, -1, :])      # (B, 64)

@torch.no_grad()
def topk_acc(model, loader, ks=(1, 3)):
    model.eval()
    correct = {k: 0 for k in ks}
    total = 0
    for x, y in loader:
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
    x_train, y_train = load_split("scenario8_dev_train.csv")
    x_val, y_val = load_split("scenario8_dev_val.csv")
    x_test, y_test = load_split("scenario8_dev_test.csv")
    print("train/val/test:", x_train.shape, x_val.shape, x_test.shape)

    train_loader = DataLoader(BeamDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(BeamDataset(x_val, y_val), batch_size=256)
    test_loader = DataLoader(BeamDataset(x_test, y_test), batch_size=256)

    model = BeamNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_top1 = 0.0
    for epoch in range(30):
        model.train()
        tot_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(y)
            n += len(y)
        acc = topk_acc(model, val_loader)
        print(f"epoch {epoch+1:2d}  train loss {tot_loss/n:.4f}  val Top-1 {acc[1]*100:.2f}%  Top-3 {acc[3]*100:.2f}%")
        if acc[1] > best_top1:
            best_top1 = acc[1]
            torch.save(model.state_dict(), "best_model.pt")

    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
    test_acc = topk_acc(model, test_loader)
    print(f"测试集（最佳模型） Top-1 {test_acc[1]*100:.2f}%  Top-3 {test_acc[3]*100:.2f}%")
    print("目标参考：官方排行榜 Top-1 60.64% / Top-3 91.08%，论文自报 Top-1 57.5% / Top-3 89.5%")

if __name__ == "__main__":
    main()
