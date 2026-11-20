# probe_mirror.py —— 主选方向 (a) 第 1 步：验证镜像映射的几何有效性（决策门）
#
# 背景：主选改进是“镜像数据增强 + 镜像 TTA”——把 216 维 LiDAR 距离向量按角度格反序，
# 标签按镜像关系重标，训练集 2831 → 5662 条；推理时原样本与镜像样本预测平均。
# 这一切成立的前提：波束编号与车辆角度之间存在单调映射，且码本中心与 LiDAR 角度窗
# 中心（75°）对齐，使得“镜像”就是“LiDAR 反序 + 标签 63-b”。本脚本验证这个前提。
#
# 检验分四部分（A、B 是真正的几何证据，C、D 是辅助核查）：
#   A. LiDAR 角度标定：用每帧最近回波估计车辆方向角，检验它与 beam_index 是否单调相关，
#      并拟合“波束→角度”直线，看码本两端角度的中点（镜像轴）是否≈75°。
#   B. 标签分布：镜像后的标签是否仍覆盖足够多的波束（不坍缩到少数几个）。
#   C. 功率向量置换检验：仅验算代码实现（把功率向量反序后 argmax 应等于 63-原标签）——
#      注意：这一步对任何置换都成立，只能证明“换对了”，不能证明“镜像关系本身是对的”，
#      真正的几何证据是 A。
#   D. SCR 角度网格自查：.mat 里存的角度列与均匀网格公式是否一致。
#
# 使用方法（在你电脑上）：
#   1. 把本文件复制到 C:\isac_project\probe_mirror.py
#   2. 打开 Anaconda Prompt → conda activate deepsense → cd C:\isac_project
#   3. python probe_mirror.py（约 1–3 分钟，读 2831 个 .mat 文件）
#   4. 把屏幕输出 + C:\isac_project\probe_mirror_results.txt 发回来
#
# 判定标准（脚本最后会直接给出 PASS / FAIL）：
#   PASS：按波束取中位后的 Spearman 相关 ≥ 0.9、单调比例 ≥ 0.8、斜率 > 0、
#         拟合镜像轴与 75° 之差 ≤ 10°、镜像标签覆盖波束数 ≥ 50
#         → 镜像映射成立，可进入 train8.py
#   FAIL：任一不满足 → 先把结果发回来一起看（可能是逐样本噪声太大或车辆角度估计
#         方法不对，不一定是镜像关系错了）；若确认是几何本身不成立，方向 (a) 作废，
#         转备选 (e) 分层分类

import os

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import spearmanr

BASE = r"C:\isac_project"
DATA_DIR = os.path.join(BASE, r"LiDAR\Scenario8\development_dataset")
CSV_PATH = os.path.join(DATA_DIR, "scenario8_dev_train.csv")
OUT_PATH = os.path.join(BASE, "probe_mirror_results.txt")

N_BINS = 216
N_BEAMS = 64
ANG_MIN = -np.pi / 6
ANG_MAX = np.pi
ANG_CENTER = (ANG_MIN + ANG_MAX) / 2


def bin_angles():
    edges = np.linspace(ANG_MIN, ANG_MAX, N_BINS + 1)
    return (edges[:-1] + edges[1:]) / 2


def load_scr(mat_path):
    data = scipy.io.loadmat(mat_path)["data"]
    if data.shape != (216, 2):
        raise ValueError(f"形状不是 (216, 2)：{data.shape}")
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def estimate_angle_min(dist, angles):
    valid = dist > 0
    if not np.any(valid):
        return np.nan
    j = int(np.argmin(np.where(valid, dist, np.inf)))
    return float(angles[j])


def main():
    angles = bin_angles()
    df = pd.read_csv(CSV_PATH)
    print(f"训练集行数：{len(df)}（期望 2831）")
    file_index = {}
    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            file_index.setdefault(name, os.path.join(root, name))
    def resolve(value):
        value = str(value)
        direct = os.path.join(DATA_DIR, value)
        if os.path.isfile(direct):
            return direct
        hit = file_index.get(os.path.basename(value))
        if hit is None:
            raise FileNotFoundError(f"找不到文件：{value}")
        return hit
    est_min, labels, powers = [], [], []
    grid_devs, skipped = [], 0
    for _, row in df.iterrows():
        try:
            dist, stored_ang = load_scr(resolve(row["unit1_lidar_SCR_1"]))
        except Exception as e:
            skipped += 1
            print(f"跳过一行（SCR 读失败）：{e}")
            continue
        ok = (dist > 0) & (stored_ang != 0)
        if np.any(ok):
            grid_devs.append(float(np.max(np.abs(stored_ang[ok] - angles[ok]))))
        est_min.append(estimate_angle_min(dist, angles))
        labels.append(int(row["beam_index_1"]) - 1)
        try:
            p = np.loadtxt(resolve(row["unit1_pwr_1"]), dtype=np.float64)
            powers.append(p if p.shape == (64,) else np.full(64, np.nan))
        except Exception:
            powers.append(np.full(64, np.nan))
    est_min = np.array(est_min)
    labels = np.array(labels)
    powers = np.array(powers)
    n = len(labels)
    print(f"成功读入：{n} 条，跳过：{skipped} 条")
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)
    out()
    out("=== D. SCR 角度网格自查（.mat 存的角度列 vs 均匀网格公式）===")
    if grid_devs:
        out(f"最大偏差中位数：{np.median(grid_devs) * 180 / np.pi:.3f}°，最大值：{np.max(grid_devs) * 180 / np.pi:.3f}°")
        out("（偏差应接近 0；若很大，说明角度网格不是均匀的，下面的标定结果不可信）")
    out()
    out("=== A. LiDAR 角度标定：车辆方向角 vs beam_index ===")
    mask = ~np.isnan(est_min)
    rho, _ = spearmanr(est_min[mask], labels[mask])
    out(f"Spearman 相关（逐样本，仅供参考）：{rho:.3f}")
    meds = np.full(N_BEAMS, np.nan)
    for b in range(N_BEAMS):
        sel = labels == b
        if np.any(sel):
            meds[b] = np.median(est_min[sel])
    present = ~np.isnan(meds)
    out(f"出现过的波束数：{int(np.sum(present))} / 64")
    d = np.diff(meds[present])
    mono = float(np.mean(d > 0)) if len(d) else 0.0
    out(f"按波束取中位后单调递增比例：{mono:.3f}（需 ≥ 0.8）")
    beams_present = np.where(present)[0]
    rho_med, _ = spearmanr(beams_present, meds[present])
    out(f"按波束取中位的 Spearman 相关：{rho_med:.3f}（判定用这个，需 ≥ 0.9）")
    slope, intercept = np.polyfit(beams_present, meds[present], 1)
    a0 = intercept
    a63 = intercept + 63 * slope
    axis = (a0 + a63) / 2
    out(f"拟合：角度 ≈ {intercept * 180 / np.pi:.1f}° + {slope * 180 / np.pi:.2f}°/波束 × beam")
    out(f"拟合的 beam 0 角度：{a0 * 180 / np.pi:.1f}°，beam 63 角度：{a63 * 180 / np.pi:.1f}°")
    out(f"拟合镜像轴：{axis * 180 / np.pi:.1f}°，与 75° 之差：{abs(axis * 180 / np.pi - 75):.1f}°（需 ≤ 10°）")
    out()
    out("=== B. 标签分布：镜像后的标签是否坍缩 ===")
    mir = 63 - labels
    out(f"原标签 distinct 数：{len(np.unique(labels))}，镜像标签 distinct 数：{len(np.unique(mir))}（需 ≥ 50）")
    vals, cnts = np.unique(labels, return_counts=True)
    top = vals[np.argsort(-cnts)[:3]]
    out(f"原标签最多的 3 个波束（0-based）：{list(top)}，个数：{sorted(cnts, reverse=True)[:3]}")
    vals_m, cnts_m = np.unique(mir, return_counts=True)
    top_m = vals_m[np.argsort(-cnts_m)[:3]]
    out(f"镜像标签最多的 3 个波束（0-based）：{list(top_m)}，个数：{sorted(cnts_m, reverse=True)[:3]}")
    out()
    out("=== C. 功率向量置换检验（仅验算实现，不证明几何）===")
    valid_p = ~np.isnan(powers).any(axis=1)
    if np.any(valid_p):
        pv = powers[valid_p]
        lab_p = labels[valid_p]
        ok_p = np.mean(np.argmax(pv[:, ::-1], axis=1) == (63 - np.argmax(pv, axis=1)))
        out(f"功率反序后 argmax 等于 63-原 argmax 的比例：{ok_p:.3f}（应为 1.000，否则是代码换错了）")
    else:
        out("无可用功率向量，跳过本项")
    out()
    out("=== 判定 ===")
    c1 = rho_med >= 0.9
    c2 = mono >= 0.8
    c3 = slope > 0
    c4 = abs(axis * 180 / np.pi - 75) <= 10
    c5 = len(np.unique(mir)) >= 50
    out(f"C1 按波束中位 Spearman ≥ 0.9：{c1}（{rho_med:.3f}）")
    out(f"C2 单调比例 ≥ 0.8：{c2}（{mono:.3f}）")
    out(f"C3 斜率 > 0：{c3}（{slope * 180 / np.pi:.2f}°/波束）")
    out(f"C4 镜像轴与 75° 之差 ≤ 10°：{c4}（{abs(axis * 180 / np.pi - 75):.1f}°）")
    out(f"C5 镜像标签 distinct 数 ≥ 50：{c5}（{len(np.unique(mir))}）")
    verdict = c1 and c2 and c3 and c4 and c5
    out()
    if verdict:
        out(" verdict: PASS —— 镜像映射成立：LiDAR 反序 + 标签 63-b，可进入 train8.py")
    else:
        out(" verdict: FAIL —— 先把结果发回来一起看：若 C1/C2 不通过，可能是逐样本噪声太大或车辆角度估计方法不对，")
        out("不一定是镜像关系错了；若 C4 不通过，说明码本中心与 75° 不对齐，简单反序不能用，")
        out("方向 (a) 作废，转备选 (e) 分层分类。")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"结果已写入：{OUT_PATH}")


if __name__ == "__main__":
    main()
