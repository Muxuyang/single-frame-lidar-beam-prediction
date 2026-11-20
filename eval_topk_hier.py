# eval_topk_hier.py —— 给分层模型（种子 0）补测 Top-2 / Top-5（不重新训练）
#
# 用途：v3 表 3 只给出了分层分类的 Top-1 / Top-3，Top-2 / Top-5 是空的。
# 本脚本只做推理、不训练、不改任何已有文件，跑完约 1 分钟。
# 模型与联合概率的算法与 train8.py 完全一致（已核对层名与 joint_log_prob）。
#
# 使用方法（在你电脑上）：
#   1. 把本文件复制到 C:\isac_project\eval_topk_hier.py（与 best_model_hier.pt 同目录）
#   2. 打开 Anaconda Prompt → conda activate deepsense → cd C:\isac_project
#   3. python eval_topk_hier.py
#   4. 先看自检行：Top-1 / Top-3 必须复现出 60.87% / 91.99%（允许 ±0.5 个点舍入差），
#      自检通过，Top-2 / Top-5 才能用；自检不通过，说明模型类定义与 train8.py 不一致，
#      不要用输出的 Top-2 / Top-5，先按下面第 0 步对齐模型定义。
#
# 第 0 步（只在自检不通过时做，别的不用改）：
#   把 train8.py 里 HierModel 类的定义原样粘贴过来，替换下面 HierModel 模板类。
#   模板按 train8.py 原文：embed = Linear(216, 64)，gru = GRU(64, 64, batch_first=True)，
#   sector_head = Linear(64, 8)，beam_head = Linear(64, 64)；forward 无激活函数；
#   联合概率 P(b) = P(s) × P(b|s)，与 train8.py 的 joint_log_prob 完全一致。
#
# 数据与标签（与 train2.py、train8.py、初稿第 2.2 节一致，不改）：
#   测试划分：C:\isac_project\LiDAR\Scenario8\development_dataset\scenario8_dev_test.csv（437 条）
#   输入：每行 unit1_lidar_SCR_1 对应的 .mat，变量名 data，取第 0 列距离（216 维），不做归一化
#   标签：beam_index_1 是 1-based，减 1 转 0-based
#   模型文件：C:\isac_project\best_model_hier.pt（只读，不覆盖；种子 0 的分层最佳模型）

import os

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\isac_project"
DATA_DIR = os.path.join(BASE, r"LiDAR\Scenario8\development_dataset")
CSV_PATH = os.path.join(DATA_DIR, "scenario8_dev_test.csv")
MODEL_PATH = os.path.join(BASE, "best_model_hier.pt")
OUT_PATH = os.path.join(BASE, "topk_hier_results.txt")

N_SECTORS = 8
BEAMS_PER_SECTOR = 8
N_BEAMS = 64

# 期望自检值：train8.py 种子 0 分层模型在 test 437 条上的已实测结果（初稿表 3 第 0 行）
EXPECTED = {"top1": 60.87, "top3": 91.99}
TOLERANCE_PP = 0.5  # 允许 ±0.5 个百分点的舍入 / 顺序差


class HierModel(nn.Module):
    """模板模型类：自检不通过时，用 train8.py 里的 HierModel 原定义替换本类。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(216, 64)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.sector_head = nn.Linear(64, N_SECTORS)
        self.beam_head = nn.Linear(64, N_BEAMS)

    def forward(self, x):
        e = self.embed(x).unsqueeze(1)       # (N, 1, 64)，单帧当作长度为 1 的序列；train8.py 原文无激活函数
        out, _ = self.gru(e)                 # (N, 1, 64)
        feat = out[:, -1, :]                 # (N, 64)
        return self.sector_head(feat), self.beam_head(feat)


def joint_log_prob(sector_logits, beam_logits):
    """返回 (N, 64) 的 log P(波束) = log P(扇区) + log P(波束|扇区)，与 train8.py 完全一致。"""
    log_ps = F.log_softmax(sector_logits, dim=1)
    log_pb = F.log_softmax(beam_logits.view(-1, N_SECTORS, BEAMS_PER_SECTOR), dim=2).view(-1, N_BEAMS)
    sector_idx = torch.arange(N_BEAMS, device=log_ps.device) // BEAMS_PER_SECTOR
    return log_ps[:, sector_idx] + log_pb


def load_scr_distance(mat_path):
    """读一个 SCR .mat，返回第 0 列距离（216 维）。列顺序已在 probe_scr.py 探明：SCR 与 raw 相反。"""
    data = scipy.io.loadmat(mat_path)["data"]
    assert data.shape == (216, 2), f"{mat_path} 形状不是 (216, 2)，实际是 {data.shape}"
    return data[:, 0].astype(np.float32)


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"测试集行数：{len(df)}（期望 437）")
    assert len(df) == 437, "测试集行数不是 437，请确认读的是 scenario8_dev_test.csv"

    # SCR .mat 的实际子目录以本地文件为准：先按 CSV 值拼路径，拼不到就按文件名在开发集内递归找。
    file_index = {}
    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            file_index.setdefault(name, os.path.join(root, name))

    def resolve_mat(value):
        value = str(value)
        direct = os.path.join(DATA_DIR, value)
        if os.path.isfile(direct):
            return direct
        hit = file_index.get(os.path.basename(value))
        if hit is None:
            raise FileNotFoundError(f"找不到 SCR 文件：{value}（已在 {DATA_DIR} 内递归查找）")
        return hit

    feats, labels = [], []
    for _, row in df.iterrows():
        mat_path = resolve_mat(row["unit1_lidar_SCR_1"])
        feats.append(load_scr_distance(mat_path))
        labels.append(int(row["beam_index_1"]) - 1)  # 1-based -> 0-based
    X = torch.tensor(np.stack(feats))              # (437, 216)
    y = torch.tensor(np.array(labels))             # (437,)

    # --- 载入模型：兼容 state_dict / 含 state_dict 键的 checkpoint / 完整模型三种存法 ---
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model = HierModel()
        model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        log_joint_all = []
        for i in range(0, len(X), 64):
            s_logits, b_logits = model(X[i:i + 64])
            log_joint_all.append(joint_log_prob(s_logits, b_logits))
        log_joint = torch.cat(log_joint_all, dim=0)  # (437, 64)，联合对数概率

    # Top-k：真值是否出现在联合概率前 k 名里
    topk_idx = torch.topk(log_joint, k=5, dim=1).indices  # (437, 5)
    results = {}
    for k in (1, 2, 3, 5):
        hit = (topk_idx[:, :k] == y.unsqueeze(1)).any(dim=1).sum().item()
        results[k] = (hit, hit / len(y) * 100)
        print(f"Top-{k}：{hit}/{len(y)} = {results[k][1]:.2f}%")

    # --- 自检：先复现已知的 Top-1 / Top-3，对不上就停，不采信 Top-2 / Top-5 ---
    ok = all(abs(results[k][1] - EXPECTED[f"top{k}"]) <= TOLERANCE_PP for k in (1, 3))
    print()
    if ok:
        print("自检通过：Top-1 / Top-3 与 train8 种子 0 已实测值一致，Top-2 / Top-5 可用。")
        print("对照基线（best_model_scr.pt）：Top-2 81.69% / Top-5 96.80%（2026-11-20 补测）。")
    else:
        print("自检不通过：Top-1 / Top-3 没复现出来，Top-2 / Top-5 不要用。")
        print("请按文件头第 0 步，把 train8.py 的 HierModel 类原样粘贴替换后重跑。")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("best_model_hier.pt（种子 0）在 scenario8_dev_test.csv（437 条）上的 Top-k 补测\n")
        for k in (1, 2, 3, 5):
            hit, pct = results[k]
            f.write(f"Top-{k}：{hit}/437 = {pct:.2f}%\n")
        f.write(f"自检（Top-1/Top-3 vs 60.87%/91.99%）：{'通过' if ok else '不通过'}\n")
    print(f"结果已写入：{OUT_PATH}")


if __name__ == "__main__":
    main()
