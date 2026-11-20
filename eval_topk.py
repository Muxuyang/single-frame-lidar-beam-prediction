# eval_topk.py —— 给已有的最佳模型补测 Top-2 / Top-5（不重新训练）
#
# 用途：初稿第 6 节局限性里写明的缺口——原脚本只输出 Top-1 / Top-3，
# 而 Jiang 等原论文自报 Top-2 80.6% / Top-5 95.6%，对照表里这两格一直空着。
# 本脚本只做推理、不训练、不改任何已有文件，跑完约 1 分钟。
#
# 使用方法（在你电脑上）：
#   1. 把本文件复制到 C:\isac_project\eval_topk.py（与 train2.py、best_model_scr.pt 同目录）
#   2. 打开 Anaconda Prompt → conda activate deepsense → cd C:\isac_project
#   3. python eval_topk.py
#   4. 先看自检行：Top-1 / Top-3 必须复现出 58.35% / 90.62%（允许 ±0.5 个点舍入差），
#      自检通过，Top-2 / Top-5 才能用；自检不通过，说明模型类定义与 train2.py 不一致，
#      不要用输出的 Top-2 / Top-5，先按下面第 0 步对齐模型定义。
#
# 第 0 步（只在自检不通过时做，别的不用改）：
#   把 train2.py 里模型类的定义原样粘贴过来，替换下面 BeamModel 模板类。
#   模板按初稿第 3.2 节写的结构：FC(216→64) + 单层 GRU(64) + FC(64→64) + Softmax，
#   单帧输入先经 FC 再 unsqueeze(1) 送入 GRU；若 train2.py 的激活函数、GRU 取值方式
#   与模板不同，以 train2.py 为准。
#   （2026-11-15 已按 train2.py 原文核对：层名 embed/gru/cls，forward 无激活函数，模板与 BeamNet 完全一致。）
#
# 数据与标签（与 train2.py、初稿第 2.2 节一致，不改）：
#   测试划分：C:\isac_project\LiDAR\Scenario8\development_dataset\scenario8_dev_test.csv（437 条）
#   输入：每行 unit1_lidar_SCR_1 对应的 .mat，变量名 data，取第 0 列距离（216 维），不做归一化
#   标签：beam_index_1 是 1-based，减 1 转 0-based
#   模型文件：C:\isac_project\best_model_scr.pt（只读，不覆盖）

import os

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn

BASE = r"C:\isac_project"
DATA_DIR = os.path.join(BASE, r"LiDAR\Scenario8\development_dataset")
CSV_PATH = os.path.join(DATA_DIR, "scenario8_dev_test.csv")
MODEL_PATH = os.path.join(BASE, "best_model_scr.pt")
OUT_PATH = os.path.join(BASE, "topk_results.txt")

# 期望自检值：train2.py 最佳模型在 test 437 条上的已实测结果（初稿第 3.5 节）
EXPECTED = {"top1": 58.35, "top3": 90.62}
TOLERANCE_PP = 0.5  # 允许 ±0.5 个百分点的舍入 / 顺序差


class BeamModel(nn.Module):
    """模板模型类：自检不通过时，用 train2.py 里的原定义替换本类。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(216, 64)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.cls = nn.Linear(64, 64)

    def forward(self, x):
        e = self.embed(x).unsqueeze(1)       # (N, 1, 64)，单帧当作长度为 1 的序列；train2.py 原文无激活函数
        out, _ = self.gru(e)                 # (N, 1, 64)
        return self.cls(out[:, -1, :])       # (N, 64)，logits


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
    # 这样无论 CSV 存的是相对路径还是纯文件名，都能找到，不用手动改路径。
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
        model = BeamModel()
        model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = []
        for i in range(0, len(X), 64):
            logits.append(model(X[i:i + 64]))
        logits = torch.cat(logits, dim=0)          # (437, 64)

    # Top-k：真值是否出现在预测前 k 名里
    topk_idx = torch.topk(logits, k=5, dim=1).indices  # (437, 5)
    results = {}
    for k in (1, 2, 3, 5):
        hit = (topk_idx[:, :k] == y.unsqueeze(1)).any(dim=1).sum().item()
        results[k] = (hit, hit / len(y) * 100)
        print(f"Top-{k}：{hit}/{len(y)} = {results[k][1]:.2f}%")

    # --- 自检：先复现已知的 Top-1 / Top-3，对不上就停，不采信 Top-2 / Top-5 ---
    ok = all(abs(results[k][1] - EXPECTED[f"top{k}"]) <= TOLERANCE_PP for k in (1, 3))
    print()
    if ok:
        print("自检通过：Top-1 / Top-3 与 train2 已实测值一致，Top-2 / Top-5 可用。")
        print("对照原论文自报：Top-2 80.6% / Top-5 95.6%（注意：原论文为 8 帧序列、原划分，")
        print("本结果为单帧、开发集 test 划分，仅作量级参考，不宣称超越。）")
    else:
        print("自检不通过：Top-1 / Top-3 没复现出来，Top-2 / Top-5 不要用。")
        print("请按文件头第 0 步，把 train2.py 的模型类原样粘贴替换 BeamModel 后重跑。")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("best_model_scr.pt 在 scenario8_dev_test.csv（437 条）上的 Top-k 补测\n")
        for k in (1, 2, 3, 5):
            hit, pct = results[k]
            f.write(f"Top-{k}：{hit}/437 = {pct:.2f}%\n")
        f.write(f"自检（Top-1/Top-3 vs 58.35%/90.62%）：{'通过' if ok else '不通过'}\n")
    print(f"结果已写入：{OUT_PATH}")


if __name__ == "__main__":
    main()
