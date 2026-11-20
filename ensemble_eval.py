# ensemble_eval.py —— 集成验证：把 3 个种子的模型概率平均，看 test 能不能再涨
#
# 背景：train9 多种子验证，baseline test Top-1 均值 58.66%/Top-3 91.46%/val 峰值均值 60.26%；
#   分层 test Top-1 均值 59.88%/Top-3 91.46%/val 峰值均值 60.30%。
#   集成（把多个模型的预测概率平均再取 Top-k）通常能再涨 1–2 个点，且不用重训。
#
# 做法（只做推理，不训练，不改模型）：
#   baseline 组：best_model_scr.pt（种子 0）+ best_model_scr_s1.pt + best_model_scr_s2.pt，
#     3 个模型的 softmax 概率平均，再算 test Top-1/Top-3 与 val Top-1；
#   分层组：best_model_hier.pt（种子 0）+ best_model_hier_s1.pt + best_model_hier_s2.pt，
#     3 个模型的联合概率（P(扇区)×P(波束|扇区)）平均，再算 test Top-1/Top-3 与 val Top-1。
#   与单模型 3 种子均值对照；baseline 与分层都做集成，才公平。
#
# 判断：
#   集成 test Top-1 比单模型均值高 ≥1 点 → 集成有效，可写进论文（两组都集成）；
#   否则集成不划算，不用。
#
# 使用方法：
#   1. 把本文件复制到 C:\isac_project\ensemble_eval.py（与 6 个模型文件同目录）
#   2. Anaconda Prompt → conda activate deepsense → cd C:\isac_project
#   3. python ensemble_eval.py（约 1–2 分钟，只做推理）
#   4. 把屏幕输出 + C:\isac_project\ensemble_results.txt 发回来

import os

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\isac_project"
DATA_DIR = os.path.join(BASE, r"LiDAR\Scenario8\development_dataset")
OUT_PATH = os.path.join(BASE, "ensemble_results.txt")

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

BASE_MODELS = ["best_model_scr.pt", "best_model_scr_s1.pt", "best_model_scr_s2.pt"]
HIER_MODELS = ["best_model_hier.pt", "best_model_hier_s1.pt", "best_model_hier_s2.pt"]


class BeamModel(nn.Module):
        """模板模型类（与 train2.py 的 BeamNet 一致）。"""
    
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
        """分层模型（train8/train9 的定义）：backbone 与 BeamModel 一致，扇区头 + 波束头。"""
    
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


def model_probs(model, X, device, kind):
        """返回 (N, 64) 的概率分布；kind='base' 用 softmax(logits)，kind='hier' 用联合概率。"""
        model.eval()
        probs = []
        with torch.no_grad():
                for i in range(0, len(X), 256):
                        xb = X[i:i + 256].to(device)
                        if kind == "base":
                                logits = model(xb)
                                probs.append(F.softmax(logits, dim=1).cpu())
                        else:
                                s_logits, b_logits = model(xb)
                                probs.append(torch.exp(joint_log_prob(s_logits, b_logits)).cpu())
        return torch.cat(probs, dim=0)


def topk_from_probs(probs, y):
        """由 (N, 64) 概率算 Top-1/Top-3。"""
        topk_idx = torch.topk(probs, k=3, dim=1).indices
        top1 = (topk_idx[:, 0] == y).float().mean().item() * 100
        top3 = (topk_idx == y.unsqueeze(1)).any(dim=1).float().mean().item() * 100
        return top1, top3


def load_models(kind, device):
        """按 kind 载入 3 个模型；缺文件时报错并列出缺哪个。"""
        names = BASE_MODELS if kind == "base" else HIER_MODELS
        cls = BeamModel if kind == "base" else HierModel
        models = []
        for name in names:
                path = os.path.join(BASE, name)
                if not os.path.isfile(path):
                        raise FileNotFoundError(f"找不到模型文件：{path}")
                m = cls().to(device)
                m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
                m.eval()
                models.append(m)
        return models


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
        out("集成验证 ensemble_eval.py：3 个种子的模型概率平均，只做推理，不训练")
        out(f"设备：{device}")
        out()
        file_index = {}
        for root, _, files in os.walk(DATA_DIR):
                for name in files:
                        file_index.setdefault(name, os.path.join(root, name))
        X_val, y_val = load_split("scenario8_dev_val.csv", file_index)
        X_test, y_test = load_split("scenario8_dev_test.csv", file_index)
        out(f"验证集：{len(y_val)} 条，测试集：{len(y_test)} 条")
        out()
        def run_ensemble(kind, label):
                """载入 3 个模型，平均概率，算 val Top-1 与 test Top-1/Top-3。返回 (val Top-1, test Top-1, test Top-3)，缺文件返回 None。"""
                try:
                        models = load_models(kind, device)
                except FileNotFoundError as e:
                        out(f"  {label}集成：跳过——{e}")
                        return None
                pv = sum(model_probs(m, X_val, device, kind) for m in models) / len(models)
                pt = sum(model_probs(m, X_test, device, kind) for m in models) / len(models)
                v1, _ = topk_from_probs(pv, y_val)
                t1, t3 = topk_from_probs(pt, y_test)
                out(f"  {label}集成：val Top-1 {v1:.2f}%，test Top-1 {t1:.2f}%，Top-3 {t3:.2f}%")
                return v1, t1, t3
        out("=== baseline 组集成（best_model_scr.pt / _s1 / _s2）===")
        base_ens = run_ensemble("base", "baseline")
        out(f"  baseline 单模型均值：val 峰值 {BASE_VAL_MEAN:.2f}%，test Top-1 {BASE_T1_MEAN:.2f}%，Top-3 {BASE_T3_MEAN:.2f}%")
        out()
        out("=== 分层组集成（best_model_hier.pt / _s1 / _s2）===")
        hier_ens = run_ensemble("hier", "分层")
        out(f"  分层单模型均值：val 峰值 {HIER_VAL_MEAN:.2f}%，test Top-1 {HIER_T1_MEAN:.2f}%，Top-3 {HIER_T3_MEAN:.2f}%")
        out()
        out("=== 结论 ===")
        out("注：集成 val 是最终模型的集成分数，单模型 val 是各轮峰值均值，两者口径略有不同；test 的比较是同口径的。")
        if base_ens is not None:
                out(f"baseline 集成 vs 单模型均值：val 变化 {base_ens[0] - BASE_VAL_MEAN:+.2f} 个点，test Top-1 变化 {base_ens[1] - BASE_T1_MEAN:+.2f} 个点，Top-3 变化 {base_ens[2] - BASE_T3_MEAN:+.2f} 个点")
        if hier_ens is not None:
                out(f"分层集成 vs 单模型均值：val 变化 {hier_ens[0] - HIER_VAL_MEAN:+.2f} 个点，test Top-1 变化 {hier_ens[1] - HIER_T1_MEAN:+.2f} 个点，Top-3 变化 {hier_ens[2] - HIER_T3_MEAN:+.2f} 个点")
        if hier_ens is not None and base_ens is not None:
                out(f"集成后分层 vs 集成后 baseline：test Top-1 相差 {hier_ens[1] - base_ens[1]:+.2f} 个点，Top-3 相差 {hier_ens[2] - base_ens[2]:+.2f} 个点")
                if hier_ens[1] - HIER_T1_MEAN >= 1.0 and hier_ens[1] - base_ens[1] > 0:
                        out("结论：集成有效——分层集成 test Top-1 比单模型均值高 ≥1 点且仍高于 baseline 集成，可写进论文（两组都集成）。")
                elif hier_ens[1] - HIER_T1_MEAN > 0:
                        out("结论：集成有提升信号，但幅度 <1 点；是否写进论文请与导师商量。")
                else:
                        out("结论：集成未见提升——不用集成，按单模型结果投稿。")
        elif hier_ens is not None:
                out("结论：baseline 组有模型文件缺失，只算了分层集成；补齐 baseline 模型后可再跑一次。")
        else:
                out("结论：两组均有模型文件缺失，未能完成集成；请确认 6 个模型文件都在 C:\\isac_project。")
        finish()


if __name__ == "__main__":
        main()
