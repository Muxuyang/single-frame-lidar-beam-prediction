# train5_multi.py —— 功率软目标（train5）补 3 个随机种子（导师要求，二1）
#
# 用途：导师要求只给 train5、train6 两组关键消融补 3 个种子，输出均值结果；
# 其余消融维持单种子，不需要全部重跑。只训练、不改 train5.py 原文。
# 本脚本与 train5.py 的模型、损失、数据、超参数完全一致（已按用户发来的原文核对），
# 只是把随机种子改成循环 0/1/2 各跑一遍。
# 种子 0 先自检：必须复现 train5 已实测的 test Top-1 54.92% / Top-3 90.85%（允许 ±0.5 个点），
# 自检通过，均值才能用；自检不通过，说明代码与 train5.py 不一致，不要用均值。
#
# 使用方法（在你电脑上）：
#   1. 把本文件复制到 C:\isac_project\train5_multi.py（与 train5.py 同目录，不覆盖它）
#   2. 打开 Anaconda Prompt → conda activate deepsense → cd C:\isac_project
#   3. python train5_multi.py（3 个种子各 30 轮，约数分钟；数据只读一次）
#   4. 看最后的“三种子均值”行和 topk 结果；结果写入 C:\isac_project\train5_multi_results.txt
#   5. 把截图发给我，我帮你写进初稿
#
# 模型与损失（train5.py 原文，不改）：
#   BeamNet：embed = Linear(216, 64)，gru = GRU(64, 64, batch_first=True)，
#   cls = Linear(64, 64)；forward 无激活函数。
#   功率软目标：功率向量（dB 负数先转线性 10^(p/10)，下限 1e-12）经温度 T=2 缩放后归一化；
#   损失 = 软交叉熵 -(p * log_softmax(logits)).sum().mean()，即用功率软目标完全替代 one-hot。
#   训练：Adam lr 1e-3，batch 64，30 轮，每轮按验证集 Top-1 保存最佳模型。
#   模型保存为 best_model_pow_s0.pt / s1 / s2（不覆盖原 train5 的 best_model_pow.pt）。
#
# 数据与标签（与 train5.py、初稿第 2.2 节一致，不改）：
#   输入：SCR .mat 第 0 列距离（216 维），不做归一化；标签 beam_index_1 减 1 转 0-based。
#   划分：train 2831 / val 775 / test 437。

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

BASE = r"C:\isac_project\LiDAR\Scenario8\development_dataset"
OUT_DIR = r"C:\isac_project"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCR_DIST_COL = 0
SEEDS = [0, 1, 2]
TOLERANCE_PP = 0.5  # 种子 0 自检允许 ±0.5 个百分点
EXPECTED_S0 = {"top1": 54.92, "top3": 90.85}  # train5 种子 0 已实测值


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


def run_seed(seed, data):
    """跑一个种子，返回（test Top-1%, Top-3%, val 峰值%）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    (x_train, y_train, p_train), (x_val, y_val, p_val), (x_test, y_test, p_test) = data
    train_loader = DataLoader(BeamDataset(x_train, y_train, p_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(BeamDataset(x_val, y_val, p_val), batch_size=256)
    test_loader = DataLoader(BeamDataset(x_test, y_test, p_test), batch_size=256)

    model = BeamNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_path = os.path.join(OUT_DIR, f"best_model_pow_s{seed}.pt")
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
        print("自检不通过：均值不要用。请确认本脚本的模型/损失与 train5.py 一致。")
        return

    mean_top1 = sum(v[0] for v in results.values()) / len(results)
    mean_top3 = sum(v[1] for v in results.values()) / len(results)
    mean_val = sum(v[2] for v in results.values()) / len(results)
    print(f"\n功率软目标（train5）三种子均值：test Top-1 {mean_top1:.2f}% / Top-3 {mean_top3:.2f}% / val 峰值均值 {mean_val:.2f}%")
    print("各种子 Top-1：" + " / ".join(f"{results[s][0]:.2f}%" for s in SEEDS))
    print("对照：基线 train2 三种子均值 test Top-1 58.66% / Top-3 91.46%。")
    with open(os.path.join(OUT_DIR, "train5_multi_results.txt"), "w", encoding="utf-8") as f:
        f.write("功率软目标（train5）三种子补测\n")
        for s in SEEDS:
            f.write(f"seed {s}：Top-1 {results[s][0]:.2f}% / Top-3 {results[s][1]:.2f}% / val 峰值 {results[s][2]:.2f}%\n")
        f.write(f"三种子均值：Top-1 {mean_top1:.2f}% / Top-3 {mean_top3:.2f}% / val 峰值均值 {mean_val:.2f}%\n")
        f.write("种子 0 自检通过（vs 54.92% / 90.85%）。\n")
    print("结果已写入：C:\\isac_project\\train5_multi_results.txt")


if __name__ == "__main__":
    main()
