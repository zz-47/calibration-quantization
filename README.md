# calibration-quantization

Token-probability calibration on real SLM logits — confidence-to-accuracy mapping, the temperature dial, quantization-aware drift, per-row vs whole-matrix scales. CPU-only, no assumptions.

---

## Studies

| # | Study | Claim to test | Status |
|---|-------|-------------|--------|
| 1 | The calibration surface | Confidence does not track accuracy — the reliability curve departs the diagonal | ✅ Complete (4 experiments measured) |
| 2 | Temperature as the calibration dial | One `T` minimizes ECE and flattens the curve; `T_cal ≠ T_ppl` | ⬜ Scaffolded |
| 3 | Quantization-aware drift | Rounding logits to 8/6/4-bit shifts the distribution — head-robust, tail-moving, widening | ⬜ Scaffolded |
| 4 | Scale granularity: per-row vs whole-matrix | Per-row scales beat whole-matrix scales; group scales saturate; output drift is amplified through the input | ⬜ Scaffolded |

---

## Repository Structure

```
calibration-quantization/
├── README.md                          ← this file
├── NOTATION.md                        ← symbol registry across all studies
├── requirements.txt                   ← pinned env
├── .venv/                             ← local environment (hidden, gitignored)
├── cq_cache/                          ← logits_135M/360M/1.7B.pt (the earlier study's cached tensors), quantized weights, drift records
├── 1_calibration_surface.ipynb        ← the raw confidence-to-accuracy mapping
├── 2_temperature_dial.ipynb           ← ECE(T), the calibration optimum
├── 3_quant_drift.ipynb                ← logit rounding: KL, top1 flips, head/tail motion
└── 4_scale_granularity.ipynb          ← per-tensor vs per-row vs per-group scales
```

---

## Study 1 — The calibration surface (complete)

**Design.** One cached forward pass of SmolLM2-135M (`Z ∈ R^(83×49152)`, float32) with the true next-token ids. Per position: confidence `c = max p`, correctness `a = [argmax p == y]`. Read the reliability diagram, ECE at `m ∈ {5, 10, 20, 40}`, the regime split, and the confidence-accuracy gap — 82 scored positions.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Confidence does not track accuracy | ECE > 0.05 | ECE = 0.1403 (m=10); mean conf 0.3191 vs mean acc 0.3171 — average flat, allocation not | ✅ Holds |
| C2 | Miscalibration is systematic | confident→over, cautious→under | reversed: conf ≥ 0.75 → acc 1.0 (8/8); mid band 0.36–0.45 → acc ~0.15 | ❌ Reversed — head underconfident, mid overconfident |
| C3 | The head dominates the instrument | shape near `c ∈ [0.1, 0.4]` | 59% of positions in conf 0.15–0.32; mean conf 0.319 ≈ `top1` | ✅ Holds |
| C4 | ECE is bin-count dependent | ECE moves with `m` | 0.077 (m=5) → 0.174 (m=40), 2.3× | ✅ Holds |

**Verdict in one line.** The model is globally calibrated (mean conf 0.319 ≈ mean acc 0.317) but locally uncalibrated (ECE 0.14), and the shape inverts the classic story — overconfident in the muddle (conf 0.36–0.45 → acc ~0.15), underconfident when it commits (conf ≥ 0.75 → acc 1.0).

---

## What this series builds toward

The 13-repo study series maps LoRA from decomposition through deployment on CPU-only real SLM weights. This repo is the trust-the-number core: study 1 reads whether `p` tracks correctness, study 2 drives the one calibration knob, study 3 measures what rounding does to the distribution, study 4 settles the scale granularity that survives quantization. Each notebook is self-contained and every number is measured on real matrices.
