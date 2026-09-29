# E170 SafeConf-v4 one-shot confirmation

All four outcome-sealed panels were opened together after Final Candidate freeze.

| Method | Utility@20 | Spearman | AURC | risk@20 | miss rate |
|---|---:|---:|---:|---:|---:|
| Ridge_USRCH | 0.215895 | 0.189082 | 0.113930 | 0.109293 | 0.693750 |
| V2_nested | 0.215895 | 0.188987 | 0.113958 | 0.108918 | 0.693750 |
| Ridge_USR | 0.201032 | 0.189722 | 0.113722 | 0.108705 | 0.697917 |
| Ridge_U | 0.183584 | 0.165006 | 0.116376 | 0.111527 | 0.714583 |
| Ridge_US | 0.183584 | 0.165006 | 0.116376 | 0.111527 | 0.714583 |
| Magnitude_raw | 0.165509 | 0.173360 | 0.116112 | 0.112103 | 0.712500 |

## Gate A

```json
{
  "delta_u20_macro": 0.050386856424903176,
  "nonnegative_strata_fraction": 0.8333333333333334,
  "risk10_relative_degradation": 0.0,
  "risk20_relative_degradation": 0.0,
  "risk50_relative_degradation": 0.005971664876681203,
  "high_risk_miss_rate_degradation": 0.0,
  "aurc_relative_degradation": 0.0,
  "valid_strata_fraction": 1.0,
  "n_planned_strata": 12,
  "gate_a_pass": true
}
```

## Paired bootstrap

- FinalV2_vs_Magnitude: ΔU20=+0.050387 [-0.026647, +0.113598], Δρ=+0.015627 [-0.027556, +0.057917].
- V1_vs_Magnitude_secondary: ΔU20=+0.035524 [-0.032667, +0.099397], Δρ=+0.016362 [-0.025923, +0.057345].
- FinalV2_vs_V1_secondary: ΔU20=+0.014863 [-0.018006, +0.041142], Δρ=-0.000736 [-0.010874, +0.009546].
- V1_vs_UniversalP_secondary: ΔU20=+0.017448 [-0.025874, +0.070735], Δρ=+0.024717 [-0.005892, +0.054416].

## Evidence boundary

- This confirms or rejects the frozen V2 on new perturbations and one held-out donor within the same study.
- It is not an external-study confirmation.
- Frozen V1 remains a secondary benchmark and cannot replace V2 after seeing this result.
- Column-unseen tasks are retained and reported; no-history failures are not removed.
