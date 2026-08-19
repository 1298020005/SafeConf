# Tahoe D1 task-cluster bootstrap postprocess

This is a postprocess-only audit. It reads an existing Tahoe SafeConf score
table and does not touch raw Tahoe parquet shards.

## Headline

- Gate: `PASS`
- aligned rho: 0.3989 (CI 0.3819 to 0.4154)
- partial rho controlling effect magnitude: 0.4534 (CI 0.4415 to 0.4669)
- RC@80 improvement: 4.27% (CI 3.95 to 4.62)
- task clusters: 8057

## Files

- `tables/TAHOE_D1_FORMAL_SUMMARY.csv`: point estimates plus task-cluster CI.
- `tables/TAHOE_D1_TASK_CLUSTER_BOOTSTRAP_CI.csv`: bootstrap intervals.
- `tables/TAHOE_D1_TASK_CLUSTER_BOOTSTRAP_DRAWS.csv`: bootstrap draw-level metrics.
- `tables/TAHOE_D1_OVERLAP_AUDIT_MANIFEST.csv`: copied G8 overlap-audit provenance.
- `TAHOE_D1_POSTPROCESS_STATUS.json`: machine-readable status.

## Gate

PASS requires the overall partial rho controlling effect magnitude to be > 0.20
and its task-cluster 95% CI lower bound to be > 0. PARTIAL means the point
estimate remains positive but the PASS rule is not met. FAIL means the point
estimate is non-positive or unusable.
