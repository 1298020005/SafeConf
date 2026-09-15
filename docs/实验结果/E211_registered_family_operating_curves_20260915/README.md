# SafeConf-Cert threshold operating curves

This directory is rebuilt only from the released E181/E182/E183 tables.

- `CERTIFIED_HIGH_TASK_CURVES.csv`: for each absolute error tolerance `tau`,
  the fraction certified high and recall among observed high-error tasks.
- `CERTIFIED_HIGH_TARGET_CURVES.csv`: the same calculation after grouping all
  technical tasks for one target; maxima preserve the lower-bound implication.
- `CERTIFICATE_GEOMETRY_SUMMARY.csv`: recomputed lower-bound tightness,
  interval width, and empirical upper coverage.
- `STATUS.json`: input lineage, numerical gates, and interpretation limits.
- `REPORT.md`: Chinese result interpretation, boundary conditions, and the
  prospective E205/E208 hand-off.

`certified_high_coverage` is the fraction of all units with `lower > tau`; it
is an issuance rate, not conformal coverage.  `lower <= tau` means `UNKNOWN`,
not safe.  Pooled rows and the default tau grid are retrospective diagnostics.
