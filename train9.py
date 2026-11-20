# train9.py —— 多种子验证：baseline 与分层模型各补种子 1、2，与种子 0 一起算 3 种子均值
#
# 背景：train8（种子 0）结果是 test Top-1 60.87%（baseline 58.35%，+2.52 点）、
#   Top-3 91.99%（+1.37 点），扇区准确率 92.91%，Oracle 66.36%；
#   但 val 峰值 59.61%（baseline 60.52%，-0.91 点）不同向——按预设标准“test 提升 ≥ 2 点
#   且 val 同向”不算成功，test 的 +2.52 点刚过 1 个标准误（约 2.4 点），可能是噪声。
#   本脚本用多种子判断这 +2.52 点是不是真的。
#
# 做法（与 train8/train2 完全相同的条件，只换种子）：
#   种子 1、2 各训一次 baseline（BeamModel，CE 损失）与一次分层模型（HierModel，联合 NLL），
#   共 4 次 30 轮训练；每次都按 val Top-1 保存最佳模型，再测 test。
#   输入 SCR 第 0 列距离（216 维），不做归一化；标签 beam_index_1 减 1；
#   Adam lr 1e-3，batch 64，30 epoch；backbone 与 train2 完全一致（无激活函数）。
#
# 保存（重要：不动原有最佳模型）：
#   只写 best_model_scr_s1.pt、best_model_scr_s2.pt、
#   best_model_hier_s1.pt、best_model_hier_s2.pt；
#   best_model_scr.pt 与 best_model_hier.pt 不会被覆盖。
#
# 判断（3 种子均值，种子 0 的已知结果已写进脚本）：
#   分层 test Top-1 均值比 baseline 高 ≥ 2 个点，且 val 峰值均值同向 → 确认改进，可以写进论文；
#   否则按脚本最后的结论行处理（未达标准 / 仍不稳 / 未证实）。
#
# 使用方法：
#   1. 把本文件复制到 C:\isac_project\train9.py（与 train8.py、best_model_scr.pt 同目录）
#   2. Anaconda Prompt → conda activate deepsense → cd C:\isac_project
#   3. python train9.py（约 10–15 分钟：4 次 30 轮训练，每次约数分钟）
#   4. 把屏幕输出 + C:\isac_project\train9_results.txt 发回来

import os

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE = r"C:\isac_project"
DATA_DIR = os.path.join(BASE, r"LiDAR\Scenario8\development_dataset")
OUT_PATH = os.path.join(BASE, "train9_results.txt")

EXPECTED = {"top1": 58.35, "top3": 90.62}
BASELINE_VAL_TOP1 = 60.52  # train2 在 val 上的峰值（epoch 16）

N_BINS = 216
N_BEAMS = 64
N_SECTORS = 8
BEAMS_PER_SECTOR = 8
SEEDS = [1, 2]
EPOCHS = 30
BATCH = 64
LR = 1e-3


class BeamModel(nn.Module):
    """模板模型类（与 eval_topk.py 相同）：自检不通过时，用 train2.py 里的原定义替换本类。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(216, 64)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.cls = nn.Linear(64, 64)

    def forward(self, x):
        e = self.embed(x).unsqueeze(1)
        out, _ = self.gru(e)
        return self.cls(out[:, -1, :])


class HierModel(nn.Module):
    """分层模型：backbone 与 BeamModel 完全一致（embed + GRU），输出改成两个头。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(216, 64)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.sector_head = nn.Linear(64, N_SECTORS)
        self.beam_head = nn.Linear(64, N_BEAMS)

    def forward(self, x):
        e = self.embed(x).unsqueeze(1)
        out, _ = self.gru(e)
        feat = out[:, -1, :]
        return self.sector_head(feat), self.beam_head(feat)


def joint_log_prob(sector_logits, beam_logits):
    """返回 (N, 64) 的 log P(波束) = log P(扇区) + log P(波束|扇区)。"""
    log_ps = F.log_softmax(sector_logits, dim=1)
    log_pb = F.log_softmax(beam_logits.view(-1, N_SECTORS, BEAMS_PER_SECTOR), dim=2).view(-1, N_BEAMS)
    sector_idx = torch.arange(N_BEAMS, device=log_ps.device) // BEAMS_PER_SECTOR
    return log_ps[:, sector_idx] + log_pb


def load_scr_distance(mat_path):
    data = scipy.io.loadmat(mat_path)["data"]
    assert data.shape == (216, 2), f"{mat_path} 形状不是 (216, 2)，实际是 {data.shape}"
    return data[:, 0].astype(np.float32)


def resolve_mat(value, file_index):
    value = str(value)
    direct = os.path.join(DATA_DIR, value)
    if os.path.isfile(direct):
        return direct
    hit = file_index.get(os.path.basename(value))
    if hit is None:
        raise FileNotFoundError(f"找不到 SCR 文件：{value}")
    return hit


def load_split(csv_name, file_index):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    feats, labels = [], []
    for _, row in df.iterrows():
        mat_path = resolve_mat(row["unit1_lidar_SCR_1"], file_index)
        feats.append(load_scr_distance(mat_path))
        labels.append(int(row["beam_index_1"]) - 1)
    return torch.tensor(np.stack(feats)), torch.tensor(np.array(labels))


def eval_beam(model, X, y, device):
    """模板模型（64 分类）的 Top-1/Top-3，用于 baseline 评估（多种子）。"""
    model.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, len(X), 64):
            logits.append(model(X[i:i + 64].to(device)).cpu())
        logits = torch.cat(logits, dim=0)
        topk_idx = torch.topk(logits, k=3, dim=1).indices
        top1 = (topk_idx[:, 0] == y).float().mean().item() * 100
        top3 = (topk_idx == y.unsqueeze(1)).any(dim=1).float().mean().item() * 100
        return top1, top3

def eval_hier(model, X, y, device):
    """分层模型的联合 Top-1/Top-3、扇区准确率、Oracle Top-1（假设扇区全对）。"""
    model.eval()
    with torch.no_grad():
        log_joint_all, sector_pred_all, oracle_hit_all = [], [], []
        for i in range(0, len(X), 256):
            xb = X[i:i + 256].to(device)
            yb = y[i:i + 256].to(device)
            s_logits, b_logits = model(xb)
            log_joint_all.append(joint_log_prob(s_logits, b_logits))
            sector_pred_all.append(torch.argmax(s_logits, dim=1))
            b8 = b_logits.view(-1, N_SECTORS, BEAMS_PER_SECTOR)
            true_s = yb // BEAMS_PER_SECTOR
            within = b8[torch.arange(len(yb), device=device), true_s]
            oracle_hit_all.append(torch.argmax(within, dim=1) == (yb % BEAMS_PER_SECTOR))
        log_joint = torch.cat(log_joint_all, dim=0)
        sector_pred = torch.cat(sector_pred_all, dim=0)
        oracle_hit = torch.cat(oracle_hit_all, dim=0)
        y_dev = y.to(device)
        topk_idx = torch.topk(log_joint, k=3, dim=1).indices
        top1 = (topk_idx[:, 0] == y_dev).float().mean().item() * 100
        top3 = (topk_idx == y_dev.unsqueeze(1)).any(dim=1).float().mean().item() * 100
        sacc = (sector_pred == (y_dev // BEAMS_PER_SECTOR)).float().mean().item() * 100
        oracle = oracle_hit.float().mean().item() * 100
        return top1, top3, sacc, oracle


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)
    def finish():
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"结果已写入：{OUT_PATH}")
    out("多种子验证 train9.py：baseline 与分层模型各补种子 1、2，与种子 0 一起算 3 种子均值")
    out(f"设备：{device}")
    out("注意：只写 best_model_scr_s1/s2.pt 与 best_model_hier_s1/s2.pt，不动 best_model_scr.pt 与 best_model_hier.pt")
    out()
    file_index = {}
    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            file_index.setdefault(name, os.path.join(root, name))
    X_train, y_train = load_split("scenario8_dev_train.csv", file_index)
    X_val, y_val = load_split("scenario8_dev_val.csv", file_index)
    X_test, y_test = load_split("scenario8_dev_test.csv", file_index)
    out(f"训练集：{len(y_train)} 条，验证集：{len(y_val)} 条，测试集：{len(y_test)} 条")
    out()
    def train_one(kind, seed):
        """kind='base' 训 BeamModel（CE 损失）；kind='hier' 训 HierModel（联合 NLL）。返回 (val 峰值, val 峰值 epoch, test Top-1, test Top-3, test 扇区准确率或 None, test Oracle 或 None)。"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if kind == "base":
            model = BeamModel().to(device)
            best_path = os.path.join(BASE, f"best_model_scr_s{seed}.pt")
        else:
            model = HierModel().to(device)
            best_path = os.path.join(BASE, f"best_model_hier_s{seed}.pt")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True)
        best_val = -1.0
        best_epoch = -1
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                if kind == "base":
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)
                else:
                    s_logits, b_logits = model(xb)
                    loss = F.nll_loss(joint_log_prob(s_logits, b_logits), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            if kind == "base":
                v1, v3 = eval_beam(model, X_val, y_val, device)
                out(f"  [{kind} seed {seed}] epoch {epoch}/{EPOCHS}  train loss {total_loss / n_batches:.4f}  val Top-1 {v1:.2f}%  Top-3 {v3:.2f}%")
            else:
                v1, v3, vsacc, _vor = eval_hier(model, X_val, y_val, device)
                out(f"  [{kind} seed {seed}] epoch {epoch}/{EPOCHS}  train loss {total_loss / n_batches:.4f}  val Top-1 {v1:.2f}%  Top-3 {v3:.2f}%  扇区准确率 {vsacc:.2f}%")
            if v1 > best_val:
                best_val = v1
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state)
        if kind == "base":
            t1, t3 = eval_beam(model, X_test, y_test, device)
            return best_val, best_epoch, t1, t3, None, None
        else:
            t1, t3, tsacc, toracle = eval_hier(model, X_test, y_test, device)
            return best_val, best_epoch, t1, t3, tsacc, toracle
    results = {}
    for seed in SEEDS:
        out(f"=== 种子 {seed}：baseline（BeamModel）===")
        results[("base", seed)] = train_one("base", seed)
        out(f"  baseline seed {seed}：val 峰值 {results[('base', seed)][0]:.2f}%（epoch {results[('base', seed)][1]}），test Top-1 {results[('base', seed)][2]:.2f}%，Top-3 {results[('base', seed)][3]:.2f}%")
        out()
        out(f"=== 种子 {seed}：分层模型（HierModel）===")
        results[("hier", seed)] = train_one("hier", seed)
        out(f"  hier seed {seed}：val 峰值 {results[('hier', seed)][0]:.2f}%（epoch {results[('hier', seed)][1]}），test Top-1 {results[('hier', seed)][2]:.2f}%，Top-3 {results[('hier', seed)][3]:.2f}%，扇区准确率 {results[('hier', seed)][4]:.2f}%，Oracle {results[('hier', seed)][5]:.2f}%")
        out()
    out("=== 汇总（含种子 0 的已知结果）===")
    base_s0 = (BASELINE_VAL_TOP1, 16, EXPECTED["top1"], EXPECTED["top3"])
    hier_s0 = (59.61, 15, 60.87, 91.99)
    def mean3(s0, s1, s2):
        return (s0 + s1 + s2) / 3.0
    base_val_mean = mean3(base_s0[0], results[("base", 1)][0], results[("base", 2)][0])
    base_t1_mean = mean3(base_s0[2], results[("base", 1)][2], results[("base", 2)][2])
    base_t3_mean = mean3(base_s0[3], results[("base", 1)][3], results[("base", 2)][3])
    hier_val_mean = mean3(hier_s0[0], results[("hier", 1)][0], results[("hier", 2)][0])
    hier_t1_mean = mean3(hier_s0[2], results[("hier", 1)][2], results[("hier", 2)][2])
    hier_t3_mean = mean3(hier_s0[3], results[("hier", 1)][3], results[("hier", 2)][3])
    out(f"baseline（种子 0/1/2）：val 峰值均值 {base_val_mean:.2f}%，test Top-1 均值 {base_t1_mean:.2f}%，Top-3 均值 {base_t3_mean:.2f}%")
    out(f"分层模型（种子 0/1/2）：val 峰值均值 {hier_val_mean:.2f}%，test Top-1 均值 {hier_t1_mean:.2f}%，Top-3 均值 {hier_t3_mean:.2f}%")
    out(f"种子 0：baseline val 峰值 {base_s0[0]:.2f}%（epoch 16），test {base_s0[2]:.2f}%/{base_s0[3]:.2f}%；分层 val 峰值 {hier_s0[0]:.2f}%（epoch 15），test {hier_s0[2]:.2f}%/{hier_s0[3]:.2f}%，扇区准确率 92.91%，Oracle 66.36%")
    out(f"种子 1：baseline val 峰值 {results[('base', 1)][0]:.2f}%（epoch {results[('base', 1)][1]}），test {results[('base', 1)][2]:.2f}%/{results[('base', 1)][3]:.2f}%；分层 val 峰值 {results[('hier', 1)][0]:.2f}%（epoch {results[('hier', 1)][1]}），test {results[('hier', 1)][2]:.2f}%/{results[('hier', 1)][3]:.2f}%")
    out(f"种子 2：baseline val 峰值 {results[('base', 2)][0]:.2f}%（epoch {results[('base', 2)][1]}），test {results[('base', 2)][2]:.2f}%/{results[('base', 2)][3]:.2f}%；分层 val 峰值 {results[('hier', 2)][0]:.2f}%（epoch {results[('hier', 2)][1]}），test {results[('hier', 2)][2]:.2f}%/{results[('hier', 2)][3]:.2f}%")
    dt1 = hier_t1_mean - base_t1_mean
    dv1 = hier_val_mean - base_val_mean
    out(f"均值对照：test Top-1 变化 {dt1:+.2f} 个点，val 峰值变化 {dv1:+.2f} 个点")
    if dt1 >= 2.0 and dv1 > 0:
        out("结论：多种子均值确认改进——test 提升 ≥ 2 个点且 val 同向，分层分类有效，可以写进论文。")
    elif dt1 > 0 and dv1 > 0:
        out("结论：test 与 val 均值都提升，但 test 提升 < 2 个点——有改进信号，未达预设的 2 点标准；是否写进论文请与导师商量。")
    elif dt1 > 0:
        out("结论：test 均值提升但 val 均值不同向——改进仍不稳，种子 0 的 test 提升很可能是噪声；不建议把分层写成创新点。")
    else:
        out("结论：多种子均值未见提升——分层分类未证实有效；按原计划投初稿 v1。")
    out("模型已保存：best_model_scr_s1.pt、best_model_scr_s2.pt、best_model_hier_s1.pt、best_model_hier_s2.pt（best_model_scr.pt 与 best_model_hier.pt 未动）")
    finish()


if __name__ == "__main__":
    main()
