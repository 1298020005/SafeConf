# v2.1 implementation repairs

1. Native control NaNs were initially confused with missing task identities. Repaired identity coverage check; official XGBoost missing-value handling retained. First attempt stopped before any fit or scoring.

2. Added missing math import; second startup had stopped before its first fit.

3. XGBoost sklearn wrapper cannot serialize estimator_type in this environment. Use the Booster serialization API, matching prior successful scripts. One startup fit was consumed; model parameters unchanged.

4. Initial remote extraction measured 526 s/250 rows. Switched to prefetch of permitted-row HDF5 chunk offsets using four HTTP workers; same role registry, cohort and data construction. Opaque cache retained; no confirmation expression materialized.

5. Added persistent HTTP sessions to Range reader to avoid a fresh TLS handshake per storage block. Existing opaque blocks/roles preserved; scientific parameters unchanged.

6. Coalesced adjacent permitted HDF5 storage blocks into at most 8 MiB HTTP requests. Same immutable cache keys and logical role guard; no extra genomic blocks requested. Restarted owned extraction, protected E208 untouched.

7. Adamson upstream train list already includes ctrl. Removed duplicate condition indexing before scoring; prepared Public vectors reused, no second source scan. Calibration proxies retain explicit control entry.

8. DEV native features were not in the TEST-only task lookup. Generated them from the same frozen train-NTC control reference using gene/state metadata; no new expression or evaluation-label fit.

9. After reviewing saved-stage resume, require the final role-scoped read receipt (not an early array file) before handoff. Predictor freeze now follows successful reload/input-isolation tests.
