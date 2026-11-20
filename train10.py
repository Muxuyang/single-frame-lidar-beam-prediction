# train10.py —— 分层改进版：8 个扇区各用一个小波束头（每组 8 选 1）
#
# 背景：train9 多种子验证，分层（train8/9 的 HierModel，1 个波束头管 64 个）test Top-1 均值
#   59.88%（baseline 58.66%，+1.22 点）、Top-3 均值 91.46%（持平）、val 峰值均值 60.30%
#   （baseline 60.26%，+0.04 点）。瓶颈在组内 8 选 1（种子 0 Oracle 66.36%，即假设扇区
#   全对也只有 66.36%），1 个头要同时分 64 个波束。
#
# 改法（只改输出头，别的全不动）：
#   backbone 与 train2 完全一致（embed 216→64、无激活函数、单层 GRU 64）；
#   扇区头 1 个（64→8）不变；波束头改成 8 个小头，每扇区 1 个（64→8），各管组内 8 选 1；
#   8 个小头参数总量 = 1 个 64 维头（8×(64×8+8) = 64×64+64 = 4160），不算“模型变大”。
#   联合概率 P(波束 b) = P(扇区 s) × P(b|s)，损失 = 扇区 CE + 扇区内波束 CE（联合训练）。
#   其余对照条件与 train8/train9 完全相同（Adam lr1e-3、batch 64、30 轮、val Top-1 选最佳）。
#
# 自检门（先跑，通不过不训练）：用 BeamModel 模板载入 best_model_scr.pt，须复现
#   test Top-1 58.35% / Top-3 90.62%（±0.5），否则模板与 train2.py 不一致。
#
# 保存（只写新文件，不动原有模型）：
#   best_model_hier2_s0.pt、best_model_hier2_s1.pt、best_model_hier2_s2.pt；
#   best_model_scr.pt、best_model_hier.pt、best_model_*_s1/s2.pt 均不动。
#
# 判断（3 种子均值，对照为 train9 的已知均值：baseline 58.66%/91.46%/val 60.26%；
#   现分层 59.88%/91.46%/val 60.30%）：
#   改进版 test Top-1 均值比现分层高 ≥1 点且 val 未明显下滑 → 有效，值得替换现分层写进论文；
#   否则按脚本最后的结论行处理。
#
# 使用方法：
#   1. 把本文件复制到 C:\isac_project\train10.py（与 best_model_scr.pt 同目录）
#   2. Anaconda Prompt → conda activate deepsense → cd C:\isac_project
#   3. python train10.py（约 8–12 分钟：自检 + 3 次 30 轮训练）
#   4. 把屏幕输出 + C:\isac_project\train10_results.txt 发回来

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
OUT_PATH = os.path.join(BASE, "train10_results.txt")

EXPECTED = {"top1": 58.35, "top3": 90.62}
BASE_T1_MEAN = 58.66
BASE_T3_MEAN = 91.46
BASE_VAL_MEAN = 60.26
HIER_T1_MEAN = 59.88
HIER_T3_MEAN = 91.46
HIER_VAL_MEAN = 60.30

N_BINS = 216
N_BEAMS = 64
N_SECTORS = 8
BEAMS_PER_SECTOR = 8
SEEDS = [0, 1, 2]
EPOCHS = 30
BATCH = 64
LR = 1e-3


class BeamModel(nn.Module):
        """模板模型类（与 train2.py 的 BeamNet 一致）：自检不通过时，用 train2.py 里的原定义替换本类。"""
    
        def __init__(self):
                super().__init__()
                self.embed = nn.Linear(216, 64)
                self.gru = nn.GRU(64, 64, batch_first=True)
                self.cls = nn.Linear(64, 64)
    
        def forward(self, x):
                e = self.embed(x).unsqueeze(1)
                out, _ = self.gru(e)
                return self.cls(out[:, -1, :])


class HierModel2(nn.Module):
        """分层改进版：backbone 与 BeamModel 完全一致（embed + GRU）；扇区头 1 个，波束头 8 个小头。"""
    
        def __init__(self):
                super().__init__()
                self.embed = nn.Linear(216, 64)
                self.gru = nn.GRU(64, 64, batch_first=True)
                self.sector_head = nn.Linear(64, N_SECTORS)
                self.beam_heads = nn.ModuleList([nn.Linear(64, BEAMS_PER_SECTOR) for _ in range(N_SECTORS)])
    
        def forward(self, x):
                e = self.embed(x).unsqueeze(1)
                out, _ = self.gru(e)
                feat = out[:, -1, :]
                beam_stack = torch.stack([h(feat) for h in self.beam_heads], dim=1)
                return self.sector_head(feat), beam_stack


def joint_log_prob(sector_logits, beam_logits):
        """返回 (N, 64) 的 log P(波束) = log P(扇区) + log P(波束|扇区)；beam_logits 形状 (N, 8, 8)。"""
        log_ps = F.log_softmax(sector_logits, dim=1)
        log_pb = F.log_softmax(beam_logits, dim=2)
        return (log_ps.unsqueeze(2) + log_pb).view(-1, N_BEAMS)


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
        """模板模型（64 分类）的 Top-1/Top-3，用于自检（baseline 评估）。"""
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


def eval_hier2(model, X, y, device):
        """分层改进版的联合 Top-1/Top-3、扇区准确率、Oracle Top-1（假设扇区全对）。"""
        model.eval()
        with torch.no_grad():
                log_joint_all, sector_pred_all, oracle_hit_all = [], [], []
                for i in range(0, len(X), 256):
                        xb = X[i:i + 256].to(device)
                        yb = y[i:i + 256].to(device)
                        s_logits, b_logits = model(xb)
                        log_joint_all.append(joint_log_prob(s_logits, b_logits))
                        sector_pred_all.append(torch.argmax(s_logits, dim=1))
                        b8 = b_logits
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
        out("分层改进版 train10.py：8 个扇区各用一个小波束头，3 种子（0/1/2），对照 train9 已知均值")
        out(f"设备：{device}")
        out("注意：只写 best_model_hier2_s0/s1/s2.pt，不动 best_model_scr.pt、best_model_hier.pt 与 best_model_*_s1/s2.pt")
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
        out("自检：用 BeamModel 模板载入 best_model_scr.pt，复现 test Top-1 58.35% / Top-3 90.62%（±0.5）")
        chk_path = os.path.join(BASE, "best_model_scr.pt")
        if not os.path.isfile(chk_path):
                out("自检失败：找不到 best_model_scr.pt（须与 train10.py 同目录）。本次不训练。")
                finish()
                return
        chk = BeamModel().to(device)
        chk_state = torch.load(chk_path, map_location=device, weights_only=False)
        chk.load_state_dict(chk_state)
        c1, c3 = eval_beam(chk, X_test, y_test, device)
        out(f"自检结果：test Top-1 {c1:.2f}%，Top-3 {c3:.2f}%")
        if abs(c1 - EXPECTED["top1"]) > 0.5 or abs(c3 - EXPECTED["top3"]) > 0.5:
                out("自检不通过：模板与 train2.py 不一致，请先核对模型定义再训练。本次不训练。")
                finish()
                return
        out("自检通过，开始训练。")
        out()
        def train_one(seed):
                """训一次 HierModel2（联合 NLL）。返回 (val 峰值, val 峰值 epoch, test Top-1, test Top-3, test 扇区准确率, test Oracle)。"""
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = HierModel2().to(device)
                best_path = os.path.join(BASE, f"best_model_hier2_s{seed}.pt")
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
                        v1, v3, vsacc, _vor = eval_hier2(model, X_val, y_val, device)
                        out(f"  [hier2 seed {seed}] epoch {epoch}/{EPOCHS}  train loss {total_loss / n_batches:.4f}  val Top-1 {v1:.2f}%  Top-3 {v3:.2f}%  扇区准确率 {vsacc:.2f}%")
                        if v1 > best_val:
                                best_val = v1
                                best_epoch = epoch
                                torch.save(model.state_dict(), best_path)
                best_state = torch.load(best_path, map_location=device, weights_only=False)
                model.load_state_dict(best_state)
                t1, t3, tsacc, toracle = eval_hier2(model, X_test, y_test, device)
                return best_val, best_epoch, t1, t3, tsacc, toracle

        results = {}
        for seed in SEEDS:
                out(f"=== 种子 {seed}：分层改进版（HierModel2）===")
                results[seed] = train_one(seed)
                out(f"  hier2 seed {seed}：val 峰值 {results[seed][0]:.2f}%（epoch {results[seed][1]}），test Top-1 {results[seed][2]:.2f}%，Top-3 {results[seed][3]:.2f}%，扇区准确率 {results[seed][4]:.2f}%，Oracle {results[seed][5]:.2f}%")
                out()
        out("=== 汇总（对照 train9 的已知 3 种子均值）===")
        def mean3(a, b, c):
                return (a + b + c) / 3.0
        h2_val_mean = mean3(results[0][0], results[1][0], results[2][0])
        h2_t1_mean = mean3(results[0][2], results[1][2], results[2][2])
        h2_t3_mean = mean3(results[0][3], results[1][3], results[2][3])
        h2_sacc_mean = mean3(results[0][4], results[1][4], results[2][4])
        h2_oracle_mean = mean3(results[0][5], results[1][5], results[2][5])
        out(f"baseline（train9 均值）：val 峰值 {BASE_VAL_MEAN:.2f}%，test Top-1 {BASE_T1_MEAN:.2f}%，Top-3 {BASE_T3_MEAN:.2f}%")
        out(f"现分层（train9 均值）：val 峰值 {HIER_VAL_MEAN:.2f}%，test Top-1 {HIER_T1_MEAN:.2f}%，Top-3 {HIER_T3_MEAN:.2f}%")
        out(f"改进版（train10 均值）：val 峰值 {h2_val_mean:.2f}%，test Top-1 {h2_t1_mean:.2f}%，Top-3 {h2_t3_mean:.2f}%，扇区准确率均值 {h2_sacc_mean:.2f}%，Oracle 均值 {h2_oracle_mean:.2f}%")
        out(f"种子 0：val 峰值 {results[0][0]:.2f}%（epoch {results[0][1]}），test {results[0][2]:.2f}%/{results[0][3]:.2f}%，扇区准确率 {results[0][4]:.2f}%，Oracle {results[0][5]:.2f}%")
        out(f"种子 1：val 峰值 {results[1][0]:.2f}%（epoch {results[1][1]}），test {results[1][2]:.2f}%/{results[1][3]:.2f}%，扇区准确率 {results[1][4]:.2f}%，Oracle {results[1][5]:.2f}%")
        out(f"种子 2：val 峰值 {results[2][0]:.2f}%（epoch {results[2][1]}），test {results[2][2]:.2f}%/{results[2][3]:.2f}%，扇区准确率 {results[2][4]:.2f}%，Oracle {results[2][5]:.2f}%")
        dt_base = h2_t1_mean - BASE_T1_MEAN
        dt_hier = h2_t1_mean - HIER_T1_MEAN
        dv_hier = h2_val_mean - HIER_VAL_MEAN
        out(f"均值对照：改进版 vs baseline，test Top-1 变化 {dt_base:+.2f} 个点；改进版 vs 现分层，test Top-1 变化 {dt_hier:+.2f} 个点，val 峰值变化 {dv_hier:+.2f} 个点")
        if dt_hier >= 1.0 and dv_hier >= -0.5:
                out("结论：分层改进版有效——test 均值比现分层高 ≥1 点且 val 未明显下滑，值得替换 train8/9 的分层写进论文。")
        elif dt_hier > 0 and dv_hier >= -0.5:
                out("结论：改进版有提升信号，但幅度 <1 点；可与现分层并列报告，或再试集成（ensemble_eval.py）。")
        elif dt_hier > 0:
                out("结论：改进版 test 均值提升但 val 下滑——改进不稳，需谨慎；保留 train8/9 的分层结果。")
        else:
                out("结论：改进版未见提升——保留 train8/9 的分层结果，不用改进版。")
        out("模型已保存：best_model_hier2_s0.pt、best_model_hier2_s1.pt、best_model_hier2_s2.pt（其余模型均未动）")
        finish()


if __name__ == "__main__":
        main()
