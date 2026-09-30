# calibration-quantization

Token-probability calibration on real SLM logits — confidence-to-accuracy mapping, the temperature dial, quantization-aware drift, per-row vs whole-matrix scales. CPU-only, no assumptions.

---

## Studies

| # | Study | Claim to test | Status |
|---|-------|-------------|--------|
| 1 | The calibration surface | Confidence does not track accuracy — the reliability curve departs the diagonal | ✅ Complete (4 experiments measured) |
| 2 | Temperature as the calibration dial | One `T` minimizes ECE and flattens the curve; `T_cal ≠ T_ppl` | ✅ Complete (4 experiments measured) |
| 3 | Quantization-aware drift | Rounding logits to 8/6/4-bit shifts the distribution — head-robust, tail-moving, widening | ✅ Complete |
| 4 | Scale granularity: per-row vs whole-matrix | Per-row scales beat whole-matrix scales; group scales saturate; output drift is amplified through the input | ✅ Complete |

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

## Study 2 — Temperature as the calibration dial (complete)

**Design.** On the same cached logits, sweep `T ∈ {0.5, 0.7, 1, 1.2, 1.5, 2, 3}`; per `T` compute `p_T = softmax(Z/T)`, ECE(m=10), and the squared gap-to-diagonal. Read the curve's minimum, the flattening at `T_cal`, and compare `T_cal` against the likelihood optimum `T_ppl` and the entropy peak `T_ent`.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | ECE has a minimum in `T` | U-shaped | 0.414 (T=0.5) → 0.139 (T=1.2) → 0.314 (T=3.0); min 1.1% below T=1; sharpening triples ECE, flattening doubles it | ✅ Holds — but the optimum is nearly flat at `T=1` and the U is asymmetric |
| C2 | `T_cal` flattens the diagram | gap at T_cal < at T=1 | gap² 0.0345 → 0.0256 (−26%); linear ECE moves only 1.1% | ⚠️ Partial — one-directional `T` trades the mid band against the head |
| C3 | Calibration ≠ likelihood optimum | `T_cal ≠ T_ppl` | 1.2 vs 1.0 (NLL minimized exactly at the training temperature) | ✅ Holds |
| C4 | Calibration ≠ entropy optimum | `T_cal ≠ T_ent` | 1.2 vs ~2 | ✅ Holds |

**Verdict in one line.** Temperature is a one-direction dial on a two-directional miscalibration — `T_cal = 1.2` fixes the overconfident mid band while worsening the underconfident head, so the scalar gain collapses to 1.1% even as the squared gap falls 26%; and the calibration optimum (1.2) is neither the likelihood optimum (1.0) nor the entropy peak (~2).

---

## Study 3 — Quantization-aware drift (complete)

**Design.** On the cached logits, quantize rows uniformly at `b ∈ {8, 6, 4}` (`s = max|z|/(2^(b−1)−1)`, zero preserved). Per depth: `KL(p_1‖p_q)`, `top1`-flip rate, `exp(H)`, and `k90` on 135M (83 rows) and 1.7B (486 rows).

**Measured findings (SmolLM2, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Rounding drift grows as bits fall | monotone in depth | KL 0.001 (8) → 0.017 (6) → 0.319 (4); flips 3.6% → 7.2% → 53% | ✅ Holds — the cliff sits at 4-bit |
| C2 | The head survives, the tail moves | few flips at all depths | 8/6-bit hold (3.6%/7.2%), 4-bit breaks (53%) | ⚠️ Partial — head-robust is bit-dependent |
| C3 | Rounding widens the distribution | exp(H) and k90 rise | exp(H) 172.9 → 153.5 (falls); k90 501 → 580 (rises) | ⚠️ Partial — tail inflates while the head sharpens |
| C4 | Drift responds to scale | sharper base drifts less | 1.7B KL 0.238 vs 0.319, flips 43% vs 53%; both land at exp(H)=153.5, k90=580 | ✅ Holds, stronger — rounding is a leveler that erases the size difference |

**Verdict in one line.** Rounding cost is a cliff, not a slope: 8-bit is nearly free (3.6% of decisions move), 4-bit replaces half the model's choices (53%) — and the coarsening is a leveler, flattening a sharper 1.7B base to the identical quantized shape as the 135M.

---

## Study 4 — Scale granularity: per-row vs whole-matrix (complete)

**Design.** Re-extract W_Q and W_gate at layers 0/10/20 (SmolLM2-135M, float32) with `X = W_E[:576]`. Quantize each at 8 and 4 bits under whole-matrix / per-row / per-group `g ∈ {8, 32}` scales. Measure weight error `‖W−W_q‖/‖W‖`, output drift `‖X·(W−W_q)ᵀ‖/‖X·Wᵀ‖`, the per-row amplification (pearson between row error and row impact), and the scale-storage cost.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Per-row beats whole-matrix | per-row < whole at both depths | gain row 2.34–4.97× across all 6 matrices and both depths; drift falls in lockstep (W_Q L0 b=4: 0.572 → 0.167) | ✅ Holds |
| C2 | Per-group refines, saturates | group gain shrinks | error strictly falls whole > g32 > g8 > row (W_Q L0 b=4: 0.616 → 0.381 → 0.274 → 0.177); marginal gains shrink; g8 never reaches per-row | ✅ Holds |
| C3 | Drift tracks error, amplified | drift/error varies by row | pearson +0.25–+0.78 (W_gate 0.41–0.78, W_Q 0.25–0.50); worst-error row ≠ worst-impact row in 7 of 12 runs | ⚠️ Partial — tracks, but the input reweights which rows matter |
| C4 | Gain is matrix-dependent | W_Q vs W_gate differ | gain row W_gate 3.70–4.97× vs W_Q 2.34–4.17×; W_Q's gain decays with depth (4.17 → 3.09 → 2.47×), W_gate holds ~4–5× | ✅ Holds — the FFN pays more for fine scales |

**Verdict in one line.** Granularity is the cheapest fidelity lever: per-row scales cut 4-bit weight error 2.3–4.8× — and the output drift with it — on every matrix; refinement saturates (whole > g32 > g8 > row); the win concentrates in the FFN; and the price is 8× the scale-storage of 8-wide groups (1536 scales vs 192).
