# NOTATION — calibration-quantization

Every symbol, metric, and named operation in this repository, defined once. Cross-references: the earlier studies defined the logit/probability objects (`Z`, `p`, `p_T`, `p_y`, `exp(H)`, `k90`, `top1`), the weight-matrix symbols (W_Q, W_gate, …), and the input rows `X = W_E[:576]`. This file defines the calibration and quantization vocabulary on top of those.

## Calibration objects

| Symbol | Meaning |
|---|---|
| `c` | confidence = `max p` at a position |
| `a` | correctness = `[argmax p == target]` (0/1) |
| `acc_b` | empirical accuracy in bin `b` |
| `c̄_b` | mean confidence in bin `b` |
| `ECE` | `Σ_b (|B_b|/N)·|c̄_b − acc_b|`, mass-weighted mean gap |
| `T` | temperature; `p_T = softmax(z/T, -1)` |
| `T_cal` | `argmin_T ECE(T)` — the calibration optimum |
| `T_ppl` | `argmin_T NLL(T)` — the likelihood optimum |
| `T_ent` | the entropy-motion peak (~2) from the earlier study's `dH/dT` |

## Quantization objects

| Symbol | Meaning |
|---|---|
| `b` | bit depth (8, 6, 4) |
| `z_q` | rounded logit row = `round(z/s)·s` |
| `s` | symmetric scale = `max\|region\| / (2^(b-1) − 1)` |
| `p_q` | `softmax(z_q, -1)` — the quantized distribution |
| `KL(p_1‖p_q)` | the drift, per position and mean |
| `top1 flip rate` | fraction of positions with `argmax p_q ≠ argmax p_1` |
| `W_q` | quantized weight matrix |
| `err_wei` | `‖W − W_q‖_F / ‖W‖_F` |
| `err_out` | `‖X·(W − W_q)ᵀ‖_F / ‖X·Wᵀ‖_F` — output drift |
| `g` | group size for per-group scales |

## Named operations

- **Reliability diagram** — `m` equal-width bins over `c`; `c̄_b` vs `acc_b` against the diagonal.
- **Regime split** — the sign of `c̄_b − acc_b` per bin (over/underconfidence).
- **Uniform symmetric rounding** — `round(z/s)·s`, zero retained exactly.
- **Granularity** — whole-matrix (one scale), per-row (one per row), per-group (one per `g` rows).
- **Trust-the-number** — the series thesis: every number a deployer spends as certainty must be measured against its referent.
