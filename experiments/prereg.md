# Pre-registration — RLA-Health (v1, 2025-10-02)

## RQs & Hypotheses
- **RQ1:** Does tri-level robustness (front-end + robust learning + robust decision/fusion) improve anomaly detection?
  **H1:** Higher F1@τ and AUPRC vs non-robust AEs/forecasters and FADE on MIT-BIH & WESAD.
- **RQ2:** Can ≤200k-param INT8 models keep ≤15 ms/window CPU latency without significant accuracy loss?
  **H2:** INT8+pruning retains ≥95% of FP32 F1 and meets latency/size targets.
- **RQ3:** Does robust multi-sensor fusion (median-of-experts + entropy-gated attention) help under sensor noise/failure?
  **H3:** Higher F1, lower false alarms/hour on WESAD vs best single modality.

## Datasets & Splits
- MIT-BIH: patient-wise disjoint; train on normal segments only; test includes arrhythmias.
- WESAD: LOSO; normal = baseline+amusement; anomaly = stress.

## Methods (fixed)
- Front-end: Hampel spike repair, STL-robust detrend, median/MAD scaling.
- Backbones: Tiny-TCN or Tiny-GRU (≤200k params).
- Losses: Huber (δ∈{1.0,1.5}), trimmed MSE (p∈{0.05,0.10}); contamination curriculum; Tukey ψ reweighting.
- Decision: s_t via robust ρ; τ = median + κ·MAD (κ∈{2.5,3.0}); temporal median smoothing.
- Fusion: median-of-experts; optional entropy-gated attention.

## Baselines & Ablations
- Baselines: IsolationForest, LOF, ARIMA-residual; LSTM/GRU-AE, Conv-AE; FADE reproduction.
- Ablations: A0 none → A1 +front-end → A2 +robust loss → A3 +decision → A4 +fusion (WESAD) → A5 +INT8 → A6 +pruning → A7 +distillation.

## Metrics & Tests
- Detection: AUROC, AUPRC, F1@τ, false alarms/hour.
- Edge: params, size (MB), CPU latency mean/p95/p99.
- Robustness: artifacts (spikes, dropouts, wander); cross-subject/device shift.
- Stats: DeLong for AUROC; 10k bootstrap CIs and paired bootstrap for AUPRC/F1.

## Hyper-grid (bounded)
- Windows L: {256,512} MIT-BIH; {128,256} WESAD; stride = L/4; H∈{1,8}.
- Model sizes: TCN channels {32,48}; GRU hidden {64,96}.
- κ ∈ {2.5, 3.0}; pruning sparsity {0.5,0.7}; QAT last 1/3 epochs.
- Early stopping: median F1 on validation; patience 15.

## Reproducibility
- Fixed seeds; deterministic ops where possible.
- Code, configs, splits, and model artifacts released with checksums.

*No test-set tuning; thresholds fit on validation normals only.*
