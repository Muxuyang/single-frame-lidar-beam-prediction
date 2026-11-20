# plot_samples.py —— 画 2 个定性样本附图（导师要求，二2）
#
# 用途：用实例支撑错误分析结论，不需要复杂绘图，控制图片尺寸，适应 WCL 4 页篇幅。
#   样本 1：典型近失错误——真值 63 号波束，模型预测第一名是 62 号（或相邻波束互换）。
#           画出该样本 216 维 LiDAR 距离向量，标注预测输出概率分布，解释相邻波束混淆。
#   样本 2：难样本——真值落在 Top-5 之外，简单分析该样本点云特征。
# 本脚本只做推理与绘图、不训练、不改任何已有文件，跑完约 1-2 分钟。
# 分析对象是基线模型（best_model_scr.pt，train2.py），与初稿第 4 节的错误分析一致。
#
# 使用方法（在你电脑上）：
#   1. 把本文件复制到 C:\isac_project\plot_samples.py
#      （与 best_model_scr.pt 同目录）
#   2. 打开 Anaconda Prompt → conda activate deepsense → cd C:\isac_project
#   3. python plot_samples.py
#   4. 先看自检行：test Top-1 必须复现出 58.35%（允许 ±0.5 个点），Top-3 复现出 90.62%。
#      自检通过，两个样本才可信；自检不通过，说明模型类定义与 train2.py 不一致，
#      先把 train2.py 里 BeamNet 类的定义原样粘贴过来，替换下面 BeamModel 模板类。
#   5. 跑完会在 C:\isac_project 下生成 sample1_nearmiss.png 和 sample2_hard.png，
#      以及 sample_summary.txt（两个样本的数字摘要：真值、预测、概率、点云特征）。
#      把这两张图和 sample_summary.txt 发给我，我帮你写进初稿。
#
# 模型模板（自检不通过时替换）：按 train2.py 原文，embed = Linear(216, 64)，
#   gru = GRU(64, 64, batch_first=True)，cls = Linear(64, 64)；forward 无激活函数。
#
# 数据与标签（与 train2.py、初稿第 2.2 节一致，不改）：
#   测试划分：C:\isac_project\LiDAR\Scenario8\development_dataset\scenario8_dev_test.csv（437 条）
#   输入：每行 unit1_lidar_SCR_1 对应的 .mat，变量名 data，取第 0 列距离（216 维），不做归一化
#   标签：beam_index_1 是 1-based，减 1 转 0-based；报告时用 1-based 波束号（与初稿一致）
#   模型文件：best_model_scr.pt，只读，不覆盖。

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\isac_project"
DATA_DIR = os.path.join(BASE, r"LiDAR\Scenario8\development_dataset")
CSV_PATH = os.path.join(DATA_DIR, "scenario8_dev_test.csv")
MODEL_PATH = os.path.join(BASE, "best_model_scr.pt")
OUT1 = os.path.join(BASE, "sample1_nearmiss.png")
OUT2 = os.path.join(BASE, "sample2_hard.png")
OUT_TXT = os.path.join(BASE, "sample_summary.txt")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9


class BeamModel(nn.Module):
    """模板模型类：自检不通过时，用 train2.py 里的 BeamNet 原定义替换本类。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(216, 64)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.cls = nn.Linear(64, 64)

    def forward(self, x):
        e = self.embed(x).unsqueeze(1)   # (N, 1, 64)；train2.py 原文无激活函数
        out, _ = self.gru(e)
        return self.cls(out[:, -1, :])


def load_scr_distance(mat_path):
    """读一个 SCR .mat，返回第 0 列距离（216 维）。列顺序已在 probe_scr.py 探明：SCR 与 raw 相反。"""
    data = scipy.io.loadmat(mat_path)["data"]
    assert data.shape == (216, 2), f"{mat_path} 形状不是 (216, 2)，实际是 {data.shape}"
    return data[:, 0].astype(np.float32)


def load_data():
    df = pd.read_csv(CSV_PATH)
    print(f"测试集行数：{len(df)}（期望 437）")
    assert len(df) == 437, "测试集行数不是 437，请确认读的是 scenario8_dev_test.csv"
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

    feats, labels, csv_idx = [], [], []
    for _, row in df.iterrows():
        mat_path = resolve_mat(row["unit1_lidar_SCR_1"])
        feats.append(load_scr_distance(mat_path))
        labels.append(int(row["beam_index_1"]) - 1)  # 1-based -> 0-based
        csv_idx.append(int(row["index"]))
    X = torch.tensor(np.stack(feats))              # (437, 216)
    y = np.array(labels)                           # (437,)
    return X, y, np.array(csv_idx), np.stack(feats)


def predict_probs(X):
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model = BeamModel()
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits_all = []
        for i in range(0, len(X), 64):
            logits_all.append(model(X[i:i + 64]))
        logits = torch.cat(logits_all, dim=0)
    return F.softmax(logits, dim=1).numpy()  # (437, 64)，每行是 64 个波束的预测概率


def beam_no(zero_based):
    """0-based 编号 -> 初稿用的 1-based 波束号。"""
    return int(zero_based) + 1


def lidar_summary(vec):
    """点云特征摘要：非零格数、距离范围、最近回波所在格。"""
    nz = int((vec > 0).sum())
    dmax = float(vec.max())
    dmin_nz = float(vec[vec > 0].min()) if nz else 0.0
    nearest_bin = int(np.argmax(vec > 0)) if nz else -1
    return nz, dmin_nz, dmax, nearest_bin


def plot_sample(vec, probs, truth, top1, title, out_path, truth_rank):
    """两栏图：左为 216 维 LiDAR 距离向量，右为 64 波束预测概率分布（真值与第一名标出）。"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    bins = np.arange(216)
    a1.plot(bins, vec, color="black", linewidth=0.8)
    a1.set_xlabel("Angle bin (0-215)")
    a1.set_ylabel("Distance (m)")
    a1.set_title("(a) 216-dim LiDAR distance vector")
    a1.set_xlim(0, 215)
    a1.spines[["top", "right"]].set_visible(False)

    # 黑白打印适配：概率柱不用灰度区分，改用纹理填充。
    # 其他波束：白底黑边；真值：黑色实心；第一名：白底斜线纹理 + 加粗黑边。
    beams = np.arange(1, 65)
    bars = a2.bar(beams, probs, color="white", edgecolor="black", linewidth=0.4)
    bars[truth].set_color("black")          # 真值：黑色实心
    bars[top1].set_color("white")           # 第一名：白底
    bars[top1].set_hatch("///")             # 第一名：斜线纹理
    bars[top1].set_edgecolor("black")
    bars[top1].set_linewidth(1.0)
    a2.axvline(truth + 1, color="black", linestyle="--", linewidth=1)
    a2.set_xlabel("Beam (1-64)")
    a2.set_ylabel("Predicted probability")
    a2.set_title(f"(b) Predicted prob. (truth rank {truth_rank})")
    a2.set_xlim(0.5, 64.5)
    a2.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    X, y, csv_idx, feats = load_data()
    probs = predict_probs(X)
    # 自检：Top-1 / Top-3 必须复现 train2 已实测值
    top1_pred = probs.argmax(axis=1)
    top3_idx = np.argsort(-probs, axis=1)[:, :3]
    top1_acc = float((top1_pred == y).mean() * 100)
    top3_acc = float(np.mean([y[i] in top3_idx[i] for i in range(len(y))]) * 100)
    print(f"自检：Top-1 {top1_acc:.2f}%（期望 58.35%）、Top-3 {top3_acc:.2f}%（期望 90.62%）")
    ok = abs(top1_acc - 58.35) <= 0.5 and abs(top3_acc - 90.62) <= 0.5
    print("自检：" + ("通过" if ok else "不通过——请按脚本顶部第 0 步对齐模型定义，不要用下面的样本"))
    if not ok:
        return

    order = np.argsort(-probs, axis=1)  # (437, 64)，每行按概率从高到低的波束下标
    rank_of_truth = np.array([int(np.where(order[i] == y[i])[0][0]) + 1 for i in range(len(y))])

    # ---- 样本 1：典型近失错误 ----
    # 优先找真值 63 号、第一名 62 号的样本（导师举的例子）；若没有，退而找任意
    # 真值排第 2 名且与第一名相邻的样本，取第一名与真值概率差距最小的那一个（最“纠结”的）。
    cand1 = []
    for i in range(len(y)):
        if rank_of_truth[i] == 2 and abs(int(top1_pred[i]) - int(y[i])) == 1:
            gap = float(probs[i, top1_pred[i]] - probs[i, y[i]])
            cand1.append((gap, i))
    cand1.sort()
    if not cand1:
        print("样本 1：没找到真值排第 2 名且与第一名相邻的样本。")
        return
    # 优先真值 63 号、第一名 62 号
    pref = [i for _, i in cand1 if beam_no(y[i]) == 63 and beam_no(top1_pred[i]) == 62]
    i1 = pref[0] if pref else cand1[0][1]
    t1, p1 = int(y[i1]), int(top1_pred[i1])
    nz1, dmin1, dmax1, nb1 = lidar_summary(feats[i1])
    print(f"\n样本 1（近失）：CSV index {csv_idx[i1]}，真值 {beam_no(t1)} 号，预测第一名 {beam_no(p1)} 号，"
          f"真值排第 2 名；P(第一名)={probs[i1, p1]:.3f}，P(真值)={probs[i1, t1]:.3f}，差距 {probs[i1, p1] - probs[i1, t1]:.3f}。"
          f"点云：非零格 {nz1}/216，距离 {dmin1:.2f}–{dmax1:.2f} m，最近回波在第 {nb1} 格。")
    plot_sample(feats[i1], probs[i1], t1, p1,
                f"Near-miss sample (truth beam {beam_no(t1)}, predicted beam {beam_no(p1)})",
                OUT1, truth_rank=2)
    print(f"  图已保存：{OUT1}")

    # ---- 样本 2：难样本（真值落在 Top-5 之外） ----
    hard = [i for i in range(len(y)) if rank_of_truth[i] > 5]
    print(f"\n难样本候选：真值在 Top-5 之外的共 {len(hard)} 条（期望约 14 条，437×3.2%）。")
    if not hard:
        print("样本 2：没有真值在 Top-5 之外的样本。")
        return
    # 取真值概率最低的那一条（模型最“看不懂”的）
    i2 = min(hard, key=lambda i: probs[i, y[i]])
    t2, p2 = int(y[i2]), int(top1_pred[i2])
    nz2, dmin2, dmax2, nb2 = lidar_summary(feats[i2])
    print(f"样本 2（难样本）：CSV index {csv_idx[i2]}，真值 {beam_no(t2)} 号，预测第一名 {beam_no(p2)} 号，"
          f"真值排第 {rank_of_truth[i2]} 名；P(第一名)={probs[i2, p2]:.3f}，P(真值)={probs[i2, t2]:.4f}。"
          f"点云：非零格 {nz2}/216，距离 {dmin2:.2f}–{dmax2:.2f} m，最近回波在第 {nb2} 格。")
    plot_sample(feats[i2], probs[i2], t2, p2,
                f"Hard sample (truth beam {beam_no(t2)} ranked {rank_of_truth[i2]})",
                OUT2, truth_rank=int(rank_of_truth[i2]))
    print(f"  图已保存：{OUT2}")

    # ---- 数字摘要（发给我写进初稿用） ----
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("定性样本摘要（基线模型 best_model_scr.pt，测试集 437 条）\n")
        f.write(f"自检：Top-1 {top1_acc:.2f}% / Top-3 {top3_acc:.2f}%（期望 58.35% / 90.62%），通过。\n\n")
        f.write(f"样本 1（近失）：CSV index {csv_idx[i1]}；真值 {beam_no(t1)} 号波束，预测第一名 {beam_no(p1)} 号波束，"
                f"真值排第 2 名；P(第一名)={probs[i1, p1]:.3f}，P(真值)={probs[i1, t1]:.3f}，"
                f"两者概率差距 {probs[i1, p1] - probs[i1, t1]:.3f}；"
                f"点云非零格 {nz1}/216，距离 {dmin1:.2f}–{dmax1:.2f} m，最近回波第 {nb1} 格。\n")
        f.write(f"样本 2（难样本）：CSV index {csv_idx[i2]}；真值 {beam_no(t2)} 号波束，预测第一名 {beam_no(p2)} 号波束，"
                f"真值排第 {rank_of_truth[i2]} 名；P(第一名)={probs[i2, p2]:.3f}，P(真值)={probs[i2, t2]:.4f}；"
                f"点云非零格 {nz2}/216，距离 {dmin2:.2f}–{dmax2:.2f} m，最近回波第 {nb2} 格。\n")
        f.write(f"难样本总数：真值在 Top-5 之外的共 {len(hard)} 条。\n")
    print(f"数字摘要已写入：{OUT_TXT}")


if __name__ == "__main__":
    main()
