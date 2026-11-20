# Single-Frame LiDAR-Aided mmWave Beam Prediction (DeepSense 6G Scenario 8)

Reproduction, error analysis, and ablation study of LiDAR-aided beam prediction,
plus an exploratory 8×8 hierarchical classification scheme.
Companion code for the paper "Single-Frame LiDAR-Aided Millimeter-Wave Beam
Prediction: Reproduction, Error Analysis, and Hierarchical Classification
Improvement" (submitted to IEEE Wireless Communications Letters).

> **Scope note.** This is a research reproduction, not an industrial codebase.
> The goal is that the core results can be reproduced from the scripts below.
> Single-frame LiDAR has a performance ceiling of about 58%–60% Top-1 on this
> task; the hierarchical scheme is an exploratory attempt (mean +1.22 points),
> not a breakthrough. Negative results are reported faithfully in the paper.

## Environment

- OS: Windows 10/11 (developed and run on Windows; scripts use `C:\isac_project` paths)
- Python 3.10.22 (Miniconda environment `deepsense`)
- PyTorch 2.6.0+cu124, torchvision 0.21.0, numpy 2.2.6, pandas 2.3.3,
  matplotlib 3.10.9, scipy 1.15.3
- GPU: NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM); a single training run takes
  a few minutes. CPU-only runs are possible but slower.

```bash
conda activate deepsense
cd C:\isac_project
python train2.py
```

> Do not double-click a `.py` file (it uses the system Python, which has no
> torch). Always run from Anaconda Prompt as above.

## Dataset

DeepSense 6G, scenario 8 (real-world V2I, Arizona State University).
Register and download at https://deepsense6g.net (free after registration).
Only the **development_dataset** (with ground-truth labels) is used;
`challenge_dataset` labels are hidden.

- Place the data at `C:\isac_project\LiDAR\Scenario8\development_dataset`
- Splits (from `scenario8_dev_train/val/test.csv`): train 2831 / val 775 / test 437
- Modalities in the downloaded dev set: 2D LiDAR (`.mat`) + mmWave power (`.txt`).
  No GPS / RGB files are present in the downloaded package (verified).
- LiDAR files: raw `(460, 2)` with column 0 = angle (rad), column 1 = distance (m);
  SCR files `(216, 2)` are the official preprocessed output, with columns reversed
  (column 0 = distance, 0–13.68 m; column 1 = angle, −2.094–1.073 rad).
- Label `beam_index_1` is 1-based (1–64); training uses `beam_index_1 − 1`.
  Verified: `argmax(power) + 1 == beam_index_1` on 100/100 samples.

## Task

Given one frame of LiDAR point cloud, predict the optimal beam among 64
(a 64-way classification). Metric: Top-k accuracy on the 437-sample test split.
Reference points (different test sets, magnitude only):
Jiang et al. self-report Top-1 57.5% / Top-3 89.5% (8-frame sequences);
official leaderboard (scenario 8, hidden test set) Top-1 60.64% / Top-3 91.08%.

## Key parameters (all experiments)

- Input: SCR 216-dim distance vector (column 0 of the SCR `.mat`), no normalization
- Model: FC embedding (216→64) + single-layer GRU (hidden 64, zero init)
  + FC (64→64) + softmax. **No activation function in the forward pass**
  (this detail matters for reproduction).
- Loss: cross-entropy. Optimizer: Adam, lr 1e-3. Batch 64. Epochs 30.
- Best model is saved by validation Top-1. Seed 0 unless stated otherwise.
- Test set n = 437, so 1 standard error ≈ 2.4 percentage points.

## Repository layout

| Script | What it does |
|---|---|
| `train2.py` | Baseline reproduction (single-frame, SCR 216-dim). Test Top-1 58.35% / Top-3 90.62%. Saves `best_model_scr.pt`. |
| `train.py` | First attempt (raw 460-dim sorted distances). Fails at 17.39% / 40.50% — cross-frame angle grids are misaligned. Kept as the failure record. |
| `train3.py` / `train4.py` / `train5.py` | Soft-label family (neighbor-kernel replace / auxiliary / power soft targets). 56.98% / 57.44% / 54.92% Top-1 — no gain over baseline. |
| `train6.py` | 1-D CNN + global average pooling. 34.78% / 63.16% — pooling discards angular position. |
| `train7.py` | Position-preserving CNN (no pooling, flatten → FC). 55.84% / 90.85% — overfits. |
| `train8.py` | Hierarchical classification, seed 0 (8 sectors × 8 beams, joint probability P(b) = P(s)·P(b\|s)). Test Top-1 60.87% / Top-3 91.99%. Saves `best_model_hier.pt`. Includes a self-check gate that reloads `best_model_scr.pt` and must reproduce 58.35% / 90.62% (±0.5 pts) before training. |
| `train9.py` | Hierarchical, seeds 1–2 (same as train8). Three-seed means: hierarchical Top-1 59.88% / Top-3 91.46% vs baseline 58.66% / 91.46%. |
| `train10.py` | Sector-specific beam heads (8 × Linear(64→8)). Three-seed mean 57.67% — worse than hierarchical. |
| `ensemble_eval.py` | Three-seed probability averaging (inference only). Hierarchical 59.73% Top-1 — no Top-1 gain. |
| `eval_topk.py` | Baseline Top-2 / Top-5 measurement (seed 0). Top-2 81.69% / Top-5 96.80%. Self-checks Top-1/Top-3 first. |
| `eval_topk_hier.py` | Hierarchical Top-2 / Top-5 measurement (seed 0). Top-2 83.30% / Top-5 96.57%. Self-checks Top-1/Top-3 first. |
| `eval_topk_hier3.py` | Hierarchical Top-2 for all three seeds (mean). |
| `plot_samples.py` | Two qualitative sample figures: a typical near-miss (truth beam 63, predicted 62) and a hard sample (truth outside Top-5). |
| `analyze_errors.py` | Error analysis on the test set (255 correct / 182 errors; 141/182 near-misses). |
| `check_data.py` / `check_all.py` / `diagnose.py` / `probe_scr.py` / `probe_time.py` | Data verification and diagnosis (label check 100/100; angle-grid misalignment up to 4.009 rad; CSV rows are not time-ordered). |
| `probe_mirror.py` / `probe_angle2.py` / `probe_files.py` | Mirror-augmentation decision-gate probes (voided: fitted mirror axis 87.9°–94.3° vs 75° window center; no GPS for calibration). |

## Reproducing the core results

1. Download the data (see Dataset) and place it under `C:\isac_project`.
2. `python train2.py` → baseline. Expect test Top-1 ≈ 58.35%, Top-3 ≈ 90.62%
   (best model saved by validation Top-1, epoch 16).
3. `python train8.py` → hierarchical, seed 0. The script first self-checks by
   reloading `best_model_scr.pt` (must reproduce 58.35% / 90.62% ±0.5 pts),
   then trains. Expect test Top-1 ≈ 60.87%, Top-3 ≈ 91.99%.
4. `python train9.py` → hierarchical, seeds 1–2 (same self-check per seed).
   Three-seed means: hierarchical Top-1 59.88% vs baseline 58.66%.
5. `python eval_topk.py` and `python eval_topk_hier.py` → Top-2 / Top-5
   (baseline 81.69% / 96.80%; hierarchical seed 0 83.30% / 96.57%).

## Main results (test split, 437 samples)

| Experiment | Top-1 | Top-3 | Note |
|---|---|---|---|
| Baseline (train2, seed 0) | 58.35% | 90.62% | Reproduction |
| Baseline Top-2 / Top-5 (seed 0) | 81.69% / 96.80% | — | eval_topk.py |
| Soft labels (train3/4/5) | 56.98% / 57.44% / 54.92% | 88.79% / 88.79% / 90.85% | No gain |
| CNN + pooling (train6) | 34.78% | 63.16% | Position lost |
| CNN flat (train7) | 55.84% | 90.85% | Overfits |
| Hierarchical (train8, seed 0) | 60.87% | 91.99% | +2.52 pts, val lower |
| Hierarchical (train9, 3-seed mean) | 59.88% | 91.46% | +1.22 pts mean, Top-3 flat |
| Hierarchical Top-2 / Top-5 (seed 0) | 83.30% / 96.57% | — | eval_topk_hier.py |
| Sector heads (train10, 3-seed mean) | 57.67% | 90.85% | Worse |
| Ensemble (hierarchical, 3 seeds) | 59.73% | 91.99% | No Top-1 gain |

## Limitations (stated in the paper)

- Except hierarchical classification (3 seeds), all ablation experiments use a
  single random seed. Test set n = 437, so metrics fluctuate by about ±2.4%.
  Large performance changes are credible; 1–2 point differences are qualitative.
- All comparisons with the paper's self-report and the leaderboard are across
  different test sets (magnitude only).
- The original paper's setting is 8-frame sequences; this work is single-frame
  because the public development set rows are not time-ordered.
- Only scenario 8 was used; scenario 9 (with radar) is left for future work.

## References

- Jiang, Charan, Alkhateeb, "LiDAR Aided Future Beam Prediction in Real-World
  Millimeter Wave V2I Communications," IEEE Wireless Communications Letters,
  vol. 12, no. 2, pp. 212–216, Feb. 2023. arXiv:2203.05548.
- Alkhateeb et al., "DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing
  and Communication Dataset," arXiv:2211.09769, 2022.
- Wu, Chakrabarti, Alkhateeb, "LiDAR-Aided Mobile Blockage Prediction in
  Real-World Millimeter Wave Systems," IEEE WCNC 2022, pp. 2631–2636
  (SCR preprocessing recipe).
