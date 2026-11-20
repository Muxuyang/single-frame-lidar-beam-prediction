# probe_angle2.py —— 主选方向 (a) 第 1 步补充检查：换个更稳的方法估计车辆角度
#
# 为什么需要这一步：probe_mirror.py 的 verdict 是 FAIL，但它的车辆角度估计器 E1
# （每帧只取最近那一个 bin 的角度）可能本身就有偏：
#   1. 车不是一个点，是个长条；最近的点可能是车头/车边，不是车中心，
#      而波束指向的是车顶天线——两者角度能差十几度；
#   2. 单个 bin 容易被随机近点（噪声尖峰、对向车道、路边物）带偏。
# 本脚本用 E3（最近一簇点）对照 E1：
#   E3 = 离最近点 2 米以内的 bin 里最长连续一段，取其中位角度。
#   一簇点（车身）比单个点稳；中位角度接近车中心，比“最近点”更接近波束指向。
# 同时保留 E1 作对照：E1 应复现 probe_mirror.py 的数字
#   （按波束中位 Spearman -0.702、轴 87.9°），复现不上说明数据或代码有出入，先停下。
# 另附：最近点距离分布（判断最近点是不是车）+ E1/E3 一致率。
# 关于 D 段那个约 90° 的偏差：它是常数坐标系偏移，不影响斜率、相关和轴差，
# 所以这里沿用同一套均匀网格，只看相对关系。
#
# 判定（看 E3）：
#   PASS：|按波束中位 Spearman| ≥ 0.9、单调主导比例 ≥ 0.8、轴与 75° 之差 ≤ 10°
#         → E1 是估计器带偏，(a) 还活着，可进 train8.py
#   FAIL-几何：关系很紧（前两项过）但轴仍差 > 10°
#         → 几何本身不支持简单镜像，(a) 作废，转备选 (e) 分层分类
#   INCONCLUSIVE：关系仍散，或两种估计器互相矛盾 → 估计器不可靠，需 GPS 或其他标定
#
# 更正：probe_mirror.py 里 C3（斜率必须 > 0）设得过严——斜率正负只说明波束编号
# 方向（码本可能是反着排的），不影响镜像成立；镜像成立只要求关系紧 + 轴对齐。
# 所以这里用 |Spearman| 和主导方向的单调比例。
#
# 使用方法（和 probe_mirror.py 一模一样）：
#   1. 把本文件复制到 C:\isac_project\probe_angle2.py
#   2. Anaconda Prompt → conda activate deepsense → cd C:\isac_project
#   3. python probe_angle2.py（约 1–3 分钟）
#   4. 把屏幕输出 + C:\isac_project\probe_angle2_results.txt 发回来

import os

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import spearmanr

BASE = r"C:\isac_project"
DATA_DIR = os.path.join(BASE, r"LiDAR\Scenario8\development_dataset")
CSV_PATH = os.path.join(DATA_DIR, "scenario8_dev_train.csv")
OUT_PATH = os.path.join(BASE, "probe_angle2_results.txt")

N_BINS = 216
N_BEAMS = 64
ANG_MIN = -np.pi / 6
ANG_MAX = np.pi


def bin_angles():
    edges = np.linspace(ANG_MIN, ANG_MAX, N_BINS + 1)
    return (edges[:-1] + edges[1:]) / 2


def load_scr(mat_path):
    data = scipy.io.loadmat(mat_path)["data"]
    if data.shape != (216, 2):
        raise ValueError(f"形状不是 (216, 2)：{data.shape}")
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def estimate_min(dist, angles):
    valid = dist > 0
    if not np.any(valid):
        return np.nan
    j = int(np.argmin(np.where(valid, dist, np.inf)))
    return float(angles[j])


def estimate_cluster(dist, angles, window=2.0):
    """最近一簇点：离最近点 window 米内的 bin 中最长连续段的中位角度。
    返回 (角度 rad, 簇长 bins, 簇平均距离 m)。"""
    valid = dist > 0
    if not np.any(valid):
        return np.nan, 0, np.nan
    dmin = float(np.min(dist[valid]))
    near = valid & (dist <= dmin + window)
    best_start, best_len, best_mean = -1, 0, np.inf
    i = 0
    while i < N_BINS:
        if not near[i]:
            i += 1
            continue
        j = i
        while j + 1 < N_BINS and near[j + 1]:
            j += 1
        length = j - i + 1
        mean_d = float(np.mean(dist[i:j + 1]))
        if length > best_len or (length == best_len and mean_d < best_mean):
            best_start, best_len, best_mean = i, length, mean_d
        i = j + 1
    if best_len == 0:
        return np.nan, 0, np.nan
    run_angles = angles[best_start:best_start + best_len]
    return float(np.median(run_angles)), int(best_len), float(best_mean)


def calibrate(est, labels):
    """对一种角度估计器做按波束标定，返回字典。"""
    mask = ~np.isnan(est)
    if not np.any(mask):
        return {"rho_sample": float("nan"), "n_beams": 0, "rho_med": float("nan"), "mono": float("nan"), "direction": "?", "slope_deg": float("nan"), "intercept_deg": float("nan"), "axis_deg": float("nan")}
    rho, _ = spearmanr(est[mask], labels[mask])
    meds = np.full(N_BEAMS, np.nan)
    for b in range(N_BEAMS):
        sel = labels == b
        if np.any(sel):
            meds[b] = np.median(est[sel])
    present = ~np.isnan(meds)
    beams_present = np.where(present)[0]
    result = {"rho_sample": float(rho), "n_beams": int(np.sum(present))}
    if len(beams_present) < 3 or float(np.std(meds[present])) == 0.0:
        result.update({"rho_med": float("nan"), "mono": float("nan"), "direction": "?", "slope_deg": float("nan"), "intercept_deg": float("nan"), "axis_deg": float("nan")})
        return result
    d = np.diff(meds[present])
    frac_up = float(np.mean(d > 0)) if len(d) else 0.0
    frac_down = float(np.mean(d < 0)) if len(d) else 0.0
    mono = max(frac_up, frac_down)
    direction = "递增" if frac_up >= frac_down else "递减"
    rho_med, _ = spearmanr(beams_present, meds[present])
    slope, intercept = np.polyfit(beams_present, meds[present], 1)
    axis = (intercept + (intercept + 63 * slope)) / 2
    result.update({"rho_med": float(rho_med), "mono": mono, "direction": direction, "slope_deg": float(slope * 180 / np.pi), "intercept_deg": float(intercept * 180 / np.pi), "axis_deg": float(axis * 180 / np.pi)})
    return result


def synthetic_test(angles):
    """合成自测：已知真映射（beam 0 → 15°，beam 63 → 134.7°，轴 75°），
    车辆为 9-bin 簇（1–2 m）+ 随机单 bin 尖峰（0.5–2 m），检验 E3 能否
    比 E1 更准地恢复映射。返回 (是否通过, 输出行列表)。"""
    rng = np.random.default_rng(0)
    true_slope = 1.9 * np.pi / 180
    true_intercept = 15.0 * np.pi / 180
    ests_min, ests_clu, labs = [], [], []
    for b in range(N_BEAMS):
        theta = true_intercept + true_slope * b
        center_bin = int(round((theta - angles[0]) / (angles[1] - angles[0])))
        for _ in range(20):
            dist = np.full(N_BINS, 25.0)
            lo = max(0, center_bin - 4)
            hi = min(N_BINS, center_bin + 5)
            dist[lo:hi] = rng.uniform(1.0, 2.0, size=hi - lo)
            spike_bin = int(rng.integers(0, N_BINS))
            dist[spike_bin] = float(rng.uniform(0.5, 2.0))
            bg = [i for i in range(N_BINS) if not (lo <= i < hi)]
            zeros = rng.choice(bg, size=10, replace=False)
            dist[zeros] = 0.0
            ests_min.append(estimate_min(dist, angles))
            a3, _, _ = estimate_cluster(dist, angles, window=2.0)
            ests_clu.append(a3)
            labs.append(b)
    cal_min = calibrate(np.array(ests_min), np.array(labs))
    cal_clu = calibrate(np.array(ests_clu), np.array(labs))
    ok = (abs(cal_clu["rho_med"]) >= 0.98 and abs(cal_clu["axis_deg"] - 75.0) <= 3.0)
    lines = ["=== 合成自测（验证 E3 实现本身没错；真值：轴 75°，斜率 +1.90°/波束）==="]
    lines.append(f"  E1（最近单点）：|Spearman| = {abs(cal_min['rho_med']):.3f}，轴 = {cal_min['axis_deg']:.1f}°")
    lines.append(f"  E3（最近一簇）：|Spearman| = {abs(cal_clu['rho_med']):.3f}，轴 = {cal_clu['axis_deg']:.1f}°")
    lines.append(f"  合成自测 E3：{'通过' if ok else '未通过（脚本有问题，先停下检查）'}")
    return ok, lines


def main():
    angles = bin_angles()
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)
    _syn_ok, syn_lines = synthetic_test(angles)
    for s in syn_lines:
        out(s)
    out()
    if not os.path.isfile(CSV_PATH):
        out("真实数据不在本机（找不到 scenario8_dev_train.csv），跳过实测部分；合成自测见上。")
        return
    df = pd.read_csv(CSV_PATH)
    out(f"训练集行数：{len(df)}（期望 2831）")
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
    est1, est3, est3b, labels = [], [], [], []
    dmins, runlens, rundists = [], [], []
    skipped = 0
    for _, row in df.iterrows():
        try:
            dist, _stored = load_scr(resolve(row["unit1_lidar_SCR_1"]))
        except Exception as e:
            skipped += 1
            print(f"跳过一行（SCR 读失败）：{e}")
            continue
        est1.append(estimate_min(dist, angles))
        a3, rl, rd = estimate_cluster(dist, angles, window=2.0)
        est3.append(a3)
        a3b, _, _ = estimate_cluster(dist, angles, window=5.0)
        est3b.append(a3b)
        labels.append(int(row["beam_index_1"]) - 1)
        valid = dist > 0
        dmins.append(float(np.min(dist[valid])) if np.any(valid) else np.nan)
        runlens.append(rl)
        rundists.append(rd)
    est1 = np.array(est1)
    est3 = np.array(est3)
    est3b = np.array(est3b)
    labels = np.array(labels)
    dmins = np.array(dmins)
    runlens = np.array(runlens)
    rundists = np.array(rundists)
    out(f"成功读入：{len(labels)} 条，跳过：{skipped} 条")
    cal1 = calibrate(est1, labels)
    cal3 = calibrate(est3, labels)
    cal3b = calibrate(est3b, labels)
    out()
    out("=== E1 对照（应复现 probe_mirror.py 的数字）===")
    out(f"逐样本 Spearman：{cal1['rho_sample']:.3f}（probe 1：-0.197）")
    out(f"按波束中位 Spearman：{cal1['rho_med']:.3f}（probe 1：-0.702）")
    out(f"单调主导比例：{cal1['mono']:.3f}（{cal1['direction']}；probe 1 报的是单调递增比例 0.421）")
    out(f"拟合：角度 ≈ {cal1['intercept_deg']:.1f}° + {cal1['slope_deg']:.2f}°/波束 × beam")
    out(f"拟合镜像轴：{cal1['axis_deg']:.1f}°，与 75° 之差：{abs(cal1['axis_deg'] - 75.0):.1f}°（probe 1：87.9°，差 12.9°）")
    repro = (abs(cal1["rho_med"] - (-0.702)) < 0.05 and abs(cal1["axis_deg"] - 87.9) < 1.0)
    out(f"E1 复现核对：{'一致' if repro else '不一致（数据或代码有出入，先停下检查）'}")
    out()
    out("=== E3 最近一簇点（窗口 2 m）===")
    out(f"逐样本 Spearman：{cal3['rho_sample']:.3f}")
    out(f"按波束中位 Spearman：{cal3['rho_med']:.3f}（判定用 |值|，需 ≥ 0.9）")
    out(f"单调主导比例：{cal3['mono']:.3f}（{cal3['direction']}，需 ≥ 0.8）")
    out(f"拟合：角度 ≈ {cal3['intercept_deg']:.1f}° + {cal3['slope_deg']:.2f}°/波束 × beam")
    out(f"拟合镜像轴：{cal3['axis_deg']:.1f}°，与 75° 之差：{abs(cal3['axis_deg'] - 75.0):.1f}°（需 ≤ 10°）")
    out(f"参考：E3（窗口 5 m）按波束中位 Spearman：{cal3b['rho_med']:.3f}，轴：{cal3b['axis_deg']:.1f}°")
    out()
    out("=== 最近点距离分布（判断最近点是不是车）===")
    out(f"dmin 中位数：{np.nanmedian(dmins):.2f} m，p10：{np.nanpercentile(dmins, 10):.2f} m，p90：{np.nanpercentile(dmins, 90):.2f} m")
    out(f"dmin < 3 m 的比例：{float(np.mean(dmins < 3.0)):.3f}（若很高，最近点常是路边固定物，不是车）")
    out(f"E3 簇长中位数：{np.nanmedian(runlens):.0f} bins，簇平均距离中位数：{np.nanmedian(rundists):.2f} m")
    both = ~np.isnan(est1) & ~np.isnan(est3)
    agree = float(np.mean(np.abs(est1[both] - est3[both]) <= 5 * np.pi / 180)) if np.any(both) else float("nan")
    out(f"E1/E3 角度一致率（|差| ≤ 5°）：{agree:.3f}（共 {int(np.sum(both))} 条）")
    out()
    out("=== 判定（看 E3）===")
    g1 = abs(cal3["rho_med"]) >= 0.9
    g2 = cal3["mono"] >= 0.8
    g3 = abs(cal3["axis_deg"] - 75.0) <= 10
    out(f"G1 |按波束中位 Spearman| ≥ 0.9：{g1}（{abs(cal3['rho_med']):.3f}）")
    out(f"G2 单调主导比例 ≥ 0.8：{g2}（{cal3['mono']:.3f}）")
    out(f"G3 镜像轴与 75° 之差 ≤ 10°：{g3}（{abs(cal3['axis_deg'] - 75.0):.1f}°）")
    out()
    if g1 and g2 and g3:
        out(" verdict: PASS —— E3 标定通过：E1 是估计器带偏，镜像映射成立（LiDAR 反序 + 标签 63-b），可进入 train8.py")
    elif g1 and g2 and not g3:
        out(" verdict: FAIL-几何 —— E3 与 E1 一致：关系很紧，但镜像轴仍偏离 75° 超过 10°，")
        out("简单镜像的几何不成立，方向 (a) 作废，转备选 (e) 分层分类。")
    else:
        out(" verdict: INCONCLUSIVE —— 关系仍不够紧，或两种估计器互相矛盾，估计器仍不可靠，")
        out("先把结果发回来一起看（可能需 GPS 标定，或直接转备选 (e)）。")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"结果已写入：{OUT_PATH}")


if __name__ == "__main__":
    main()
