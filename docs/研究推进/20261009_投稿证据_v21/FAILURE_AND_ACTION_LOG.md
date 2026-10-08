# v2.1 implementation repairs

1. Native control NaNs were initially confused with missing task identities. Repaired identity coverage check; official XGBoost missing-value handling retained. First attempt stopped before any fit or scoring.

2. Added missing math import; second startup had stopped before its first fit.

3. XGBoost sklearn wrapper cannot serialize estimator_type in this environment. Use the Booster serialization API, matching prior successful scripts. One startup fit was consumed; model parameters unchanged.

4. Initial remote extraction measured 526 s/250 rows. Switched to prefetch of permitted-row HDF5 chunk offsets using four HTTP workers; same role registry, cohort and data construction. Opaque cache retained; no confirmation expression materialized.
