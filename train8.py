# train8.py —— 备选 (e) 分层分类：8 扇区粗分 + 扇区内 8 选 1 的联合概率模型
#
# 背景：主选 (a) 镜像增强已作废——probe_mirror.py FAIL、probe_angle2.py INCONCLUSIVE、
#   本地查不到 GPS 文件（probe_files.py：开发集只有 LiDAR + 功率，没有 GPS/RGB，
#   连 unit2 文件夹都没有），简单镜像的几何验证走不通。
#   原计划 train8.py 是镜像增强，现改为备选 (e)。初稿 v1 不动；
#   本实验只对照 train2 的 baseline（test Top-1 58.35% / Top-3 90.62%）。
#
# 方法（只改输出结构，其余与 train2 完全相同）：
#   backbone 与 train2 完全一致：FC(216→64) + ReLU + 单层 GRU(64) + 取最后一步，
#   得到 64 维特征；然后分两个头：
#     扇区头 Linear(64→8)：判断车辆在哪个扇区；
#     波束头 Linear(64→64)：64 个波束的 logits，按扇区分成 8 组、每组 8 个。
#   扇区划分（0-based 波束号）：0-7→扇区0，8-15→扇区1，……，56-63→扇区7。
#   联合概率：P(波束 b) = P(扇区 s) × P(b | s)，其中 P(b|s) 是波束头在扇区 s
#   的 8 个 logits 上做的 softmax。损失 = -log P(真值波束)
#   = 扇区 CE + 扇区内波束 CE（联合训练，不是先硬分扇区再分波束；
#   推理时 8 个扇区的概率都参与，不是一锤子定扇区）。
#   针对的正是错误分析里的结构：182 个错里 141 个（77.5%）真值排第 2–3 名，
#   且集中在相邻的 62/63 号波束——模型分不清“附近”波束，先定扇区再定波束
#   就是帮它先把“大致方向”定下来。
#
# 对照条件（与 train2 完全相同，只改输出结构/损失）：
#   输入 SCR 第 0 列距离（216 维），不做归一化；标签 beam_index_1 减 1；
#   train 2831 / val 775 / test 437；Adam lr 1e-3，batch 64，30 epoch，种子 0；
#   每 epoch 在 val 上算联合 Top-1/Top-3，取 val Top-1 最高者为最佳模型。
#
# 自检门（先验 backbone 与 train2 一致，再训分层模型）：
#   脚本先用模板 BeamModel（与 eval_topk.py 相同）载入 best_model_scr.pt，
#   在 test 上复现 Top-1 58.35% / Top-3 90.62%（允许 ±0.5 个点）。
#   自检不通过 → 停止，不训分层模型；按文件头第 0 步对齐模型定义。
#   自检通过 → 从头训练分层模型（不用 checkpoint 的权重做初始化，保证与 train2
#   是干净的同条件对照）。
#
# 第 0 步（只在自检不通过时做）：
#   把 train2.py 里模型类的定义原样粘贴过来，替换下面 BeamModel 模板类。
#   （2026-11-15 已按 train2.py 原文核对：层名 embed/gru/cls，forward 无激活函数，模板与 BeamNet 完全一致。）
#
# 成功标准（与镜像方向相同）：test Top-1 比 baseline 高 ≥ 2 个点，且 val 同向；
#   差距 < 2.4 个点（test 437 条，1 个标准误）视为噪声带，未证实改进。
#
# 使用方法：
#   1. 把本文件复制到 C:\isac_project\train8.py（与 train2.py、best_model_scr.pt 同目录）
#   2. Anaconda Prompt → conda activate deepsense → cd C:\isac_project
#   3. python train8.py（约数分钟：先自检约 1 分钟，再训 30 轮）
#   4. 把屏幕输出 + C:\isac_project\train8_results.txt 发回来

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
MODEL_PATH = os.path.join(BASE, "best_model_scr.pt")
BEST_PATH = os.path.join(BASE, "best_model_hier.pt")
OUT_PATH = os.path.join(BASE, "train8_results.txt")

EXPECTED = {"top1": 58.35, "top3": 90.62}
TOLERANCE_PP = 0.5
BASELINE_VAL_TOP1 = 60.52  # train2 在 val 上的峰值（epoch 16）

N_BINS = 216
N_BEAMS = 64
N_SECTORS = 8
BEAMS_PER_SECTOR = 8
SEED = 0
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


def eval_beam(model, X, y):
    """模板模型（64 分类）的 Top-1/Top-3，用于自检门。"""
    model.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, len(X), 64):
            logits.append(model(X[i:i + 64]))
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
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)
    def finish():
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"结果已写入：{OUT_PATH}")
    out("分层分类 train8.py：8 扇区 × 扇区内 8 选 1，联合概率模型")
    out("扇区划分：波束 0-7→扇区0，8-15→扇区1，……，56-63→扇区7")
    out(f"设备：{device}")
    out()
    out("=== 自检门：先复现 train2 的 baseline，确认 backbone 一致 ===")
    if not os.path.isfile(MODEL_PATH):
        out(f"找不到 {MODEL_PATH}，自检无法做；请确认 best_model_scr.pt 在 C:\\isac_project 下。")
        finish()
        return
    file_index = {}
    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            file_index.setdefault(name, os.path.join(root, name))
    X_test, y_test = load_split("scenario8_dev_test.csv", file_index)
    out(f"测试集行数：{len(y_test)}（期望 437）")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(ckpt, nn.Module):
        beam_model = ckpt
    else:
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        beam_model = BeamModel()
        beam_model.load_state_dict(state)
    beam_model.eval()
    t1, t3 = eval_beam(beam_model, X_test, y_test)
    out(f"自检：best_model_scr.pt 在 test 上 Top-1 {t1:.2f}%（期望 58.35%），Top-3 {t3:.2f}%（期望 90.62%）")
    ok = abs(t1 - EXPECTED["top1"]) <= TOLERANCE_PP and abs(t3 - EXPECTED["top3"]) <= TOLERANCE_PP
    out(f"自检：{'通过' if ok else '不通过'}")
    if not ok:
        out("自检不通过：不要继续训练分层模型。")
        out("请按文件头第 0 步，把 train2.py 的模型类原样粘贴替换 BeamModel 后重跑。")
        finish()
        return
    out("自检通过：backbone 与 train2 一致，开始从头训练分层模型。")
    out()
    X_train, y_train = load_split("scenario8_dev_train.csv", file_index)
    X_val, y_val = load_split("scenario8_dev_val.csv", file_index)
    out(f"训练集：{len(y_train)} 条，验证集：{len(y_val)} 条")
    model = HierModel().to(device)
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
            s_logits, b_logits = model(xb)
            loss = F.nll_loss(joint_log_prob(s_logits, b_logits), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        v1, v3, vsacc, _vor = eval_hier(model, X_val, y_val, device)
        out(f"epoch {epoch}/{EPOCHS}  train loss {total_loss / n_batches:.4f}  val Top-1 {v1:.2f}%  Top-3 {v3:.2f}%  扇区准确率 {vsacc:.2f}%")
        if v1 > best_val:
            best_val = v1
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_PATH)
            out(f"  保存最佳模型（val Top-1 {best_val:.2f}%，epoch {epoch}）")
    out()
    best_state = torch.load(BEST_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_state)
    test_t1, test_t3, test_sacc, test_oracle = eval_hier(model, X_test, y_test, device)
    out(f"=== test 集（最佳模型，epoch {best_epoch}）===")
    out(f"Top-1：{test_t1:.2f}%（baseline 58.35%），Top-3：{test_t3:.2f}%（baseline 90.62%）")
    out(f"扇区准确率（test）：{test_sacc:.2f}%（扇区 = 波束 // 8，8 分类）")
    out(f"Oracle Top-1（test，假设扇区全对）：{test_oracle:.2f}%（扇区内 8 选 1 本身的难度）")
    dt1 = test_t1 - EXPECTED["top1"]
    dv1 = best_val - BASELINE_VAL_TOP1
    out(f"与 baseline（train2）对照：test Top-1 变化 {dt1:+.2f} 个点，val Top-1 变化 {dv1:+.2f} 个点（val baseline 峰值 60.52%）")
    if dt1 >= 2.0 and dv1 > 0:
        out("达到预设成功标准：test 提升 ≥ 2 个点且 val 同向，分层分类有效。")
    elif abs(dt1) < 2.4:
        out("在噪声带内（test 437 条，1 个标准误约 2.4 个点）：未证实改进。")
    elif dt1 > 0:
        out("test 提升够了但 val 不同向：test/val 不一致，改进不稳。")
    else:
        out("低于 baseline：分层分类未起效。")
    out(f"最佳模型已保存：{BEST_PATH}")
    finish()


if __name__ == "__main__":
    main()
