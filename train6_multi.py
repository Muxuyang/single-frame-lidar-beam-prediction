# train6_multi.py —— 一维 CNN + 全局池化（train6）补 3 个随机种子（导师要求，二1）
#
# 用途：导师要求只给 train5、train6 两组关键消融补 3 个种子，输出均值结果；
# 其余消融维持单种子，不需要全部重跑。只训练、不改 train6.py 原文。
# 本脚本与 train6.py 的模型、损失、数据、超参数完全一致（已按用户发来的原文核对），
# 只是把随机种子改成循环 0/1/2 各跑一遍。
# 种子 0 先自检：必须复现 train6 已实测的 test Top-1 34.78% / Top-3 63.16%（允许 ±0.5 个点），
# 自检通过，均值才能用；自检不通过，说明代码与 train6.py 不一致，不要用均值。
#
# 使用方法（在你电脑上）：
#   1. 把本文件复制到 C:\isac_project\train6_multi.py（与 train6.py 同目录，不覆盖它）
#   2. 打开 Anaconda Prompt → conda activate deepsense → cd C:\isac_project
#   3. python train6_multi.py（3 个种子各 30 轮，约数分钟；数据只读一次）
#   4. 看最后的“三种子均值”行；结果写入 C:\isac_project\train6_multi_results.txt
#   5. 把截图发给我，我帮你写进初稿
#
# 模型与损失（train6.py 原文，不改）：
#   BeamCNN：一维卷积 Conv1d(1,32,k=5,pad=2) + ReLU + MaxPool1d(2)，
#   Conv1d(32,64,k=5,pad=2) + ReLU + MaxPool1d(2)，
#   Conv1d(64,64,k=3,pad=1) + ReLU + AdaptiveAvgPool1d(1)，再 Linear(64,64) 分类。
#   损失 = 标准交叉熵。训练：Adam lr 1e-3，batch 64，30 轮，每轮按验证集 Top-1 保存最佳模型。
#   模型保存为 best_model_cnn_s0.pt / s1 / s2（不覆盖原 train6 的 best_model_cnn.pt）。
#
# 数据与标签（与 train6.py、初稿第 2.2 节一致，不改）：
#   输入：SCR .mat 第 0 列距离（216 维），不做归一化；标签 beam_index_1 减 1 转 0-based。
#   划分：train 2831 / val 775 / test 437。

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

BASE = r"C:\isac_project\LiDAR\Scenario8\development_dataset"
OUT_DIR = r"C:\isac_project"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCR_DIST_COL = 0
SEEDS = [0, 1, 2]
TOLERANCE_PP = 0.5  # 种子 0 自检允许 ±0.5 个百分点
EXPECTED_S0 = {"top1": 34.78, "top3": 63.16}  # train6 种子 0 已实测值


def load_split(csv_name):
    df = pd.read_csv(os.path.join(BASE, csv_name))
    xs, ys = [], []
    for _, row in df.iterrows():
        d = loadmat(os.path.join(BASE, row["unit1_lidar_SCR_1"]))["data"]
        xs.append(d[:, SCR_DIST_COL].astype(np.float32))
        ys.append(int(row["beam_index_1"]) - 1)
    return np.stack(xs), np.array(ys)


class BeamDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class BeamCNN(nn.Module):
    """一维CNN：卷积核看相邻角度格的局部结构，再全局池化+分类"""

    def __init__(self, n_beam=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cls = nn.Linear(64, n_beam)

    def forward(self, x):
        f = self.features(x.unsqueeze(1)).squeeze(-1)
        return self.cls(f)


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


def run_seed(seed, data):
    """跑一个种子，返回（test Top-1%, Top-3%, val 峰值%）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    train_loader = DataLoader(BeamDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(BeamDataset(x_val, y_val), batch_size=256)
    test_loader = DataLoader(BeamDataset(x_test, y_test), batch_size=256)

    model = BeamCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    save_path = os.path.join(OUT_DIR, f"best_model_cnn_s{seed}.pt")
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
        print(f"  seed {seed} epoch {epoch+1:2d}  train loss {tot_loss/n:.4f}  val Top-1 {acc[1]*100:.2f}%  Top-3 {acc[3]*100:.2f}%")
        if acc[1] > best_top1:
            best_top1 = acc[1]
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    test_acc = topk_acc(model, test_loader)
    print(f"  seed {seed} 测试集（最佳模型） Top-1 {test_acc[1]*100:.2f}%  Top-3 {test_acc[3]*100:.2f}%  val 峰值 {best_top1*100:.2f}%")
    return test_acc[1] * 100, test_acc[3] * 100, best_top1 * 100


def main():
    print("设备:", DEVICE)
    print("加载数据（只读一次，三种子共用）...")
    data = (
        load_split("scenario8_dev_train.csv"),
        load_split("scenario8_dev_val.csv"),
        load_split("scenario8_dev_test.csv"),
    )
    print("train/val/test:", data[0][0].shape, data[1][0].shape, data[2][0].shape)

    results = {}
    for seed in SEEDS:
        print(f"\n=== 种子 {seed} ===")
        results[seed] = run_seed(seed, data)

    s0 = results[0]
    ok = abs(s0[0] - EXPECTED_S0["top1"]) <= TOLERANCE_PP and abs(s0[1] - EXPECTED_S0["top3"]) <= TOLERANCE_PP
    print(f"\n种子 0 自检（Top-1/Top-3 vs {EXPECTED_S0['top1']}%/{EXPECTED_S0['top3']}%）：{'通过' if ok else '不通过'}")
    if not ok:
        print("自检不通过：均值不要用。请确认本脚本的模型/损失与 train6.py 一致。")
        return

    mean_top1 = sum(v[0] for v in results.values()) / len(results)
    mean_top3 = sum(v[1] for v in results.values()) / len(results)
    mean_val = sum(v[2] for v in results.values()) / len(results)
    print(f"\n一维 CNN + 全局池化（train6）三种子均值：test Top-1 {mean_top1:.2f}% / Top-3 {mean_top3:.2f}% / val 峰值均值 {mean_val:.2f}%")
    print("各种子 Top-1：" + " / ".join(f"{results[s][0]:.2f}%" for s in SEEDS))
    print("对照：基线 train2 三种子均值 test Top-1 58.66% / Top-3 91.46%。")
    with open(os.path.join(OUT_DIR, "train6_multi_results.txt"), "w", encoding="utf-8") as f:
        f.write("一维 CNN + 全局池化（train6）三种子补测\n")
        for s in SEEDS:
            f.write(f"seed {s}：Top-1 {results[s][0]:.2f}% / Top-3 {results[s][1]:.2f}% / val 峰值 {results[s][2]:.2f}%\n")
        f.write(f"三种子均值：Top-1 {mean_top1:.2f}% / Top-3 {mean_top3:.2f}% / val 峰值均值 {mean_val:.2f}%\n")
        f.write("种子 0 自检通过（vs 34.78% / 63.16%）。\n")
    print("结果已写入：C:\\isac_project\\train6_multi_results.txt")


if __name__ == "__main__":
    main()
