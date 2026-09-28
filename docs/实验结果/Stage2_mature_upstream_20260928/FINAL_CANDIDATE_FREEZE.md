# SafeConf Final Candidate Freeze

Final Candidate: **V2_nested_evidence_shrinkage**.

This decision was made from DEV/SEEN only. No SEALED feature distribution, prediction, truth or result was opened.

## Registered aggregate development gate

- `delta_u20_macro`: `0.009711226587606037`
- `nonnegative_strata_fraction`: `0.875`
- `risk10_relative_degradation`: `0.0177281386903658`
- `risk20_relative_degradation`: `0.0`
- `risk50_relative_degradation`: `0.0`
- `high_risk_miss_rate_degradation`: `0.0`
- `aurc_relative_degradation`: `0.0`
- `valid_strata_fraction`: `1.0`
- `n_evaluation_strata`: `8`

## Interpretation

V2 passed the preregistered practical gate across both TxPert architectures. The paired bootstrap interval for its incremental Utility@20 still crosses zero, so this freeze selects the confirmation candidate; it does not assert independent confirmation.

Quality is absent rather than imputed. V2 uses Support, Relevance, a Conflict proxy, Content and Missingness, with monotonic evidence weights.
