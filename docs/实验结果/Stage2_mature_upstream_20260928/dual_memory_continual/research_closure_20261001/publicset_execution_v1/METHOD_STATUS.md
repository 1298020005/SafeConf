# Current scientific decisions

The user requires experiments and final-model selection before manuscript work. Existing paper assets are preserved, but writing/compilation/packaging is stopped.

- DeepSets: no further architecture expansion. Its extra set context has not established an increment over the matched pointwise network.
- Public: do not automatically choose the old HGB builder. On the same McFaline DEV cohort, simple same-context support weighting gives U20 0.872175, old HGB 0.827009, pointwise 0.869870 and DeepSets 0.864788. All remain in the matched comparison.
- Source error learning: finalize the already-registered context-scoped HGB comparison and the quantity/real-content/physical-content-null controls before choosing its role. Source teachers exclude the evaluation context's treated data; the risk training population uses the same context and outer-train genes only.
- Target feedback: freeze the final Public/risk versions first; rerun only affected combinations under the existing feedback split and error-label budgets.

The fixed-public-distance results do not decide the complete Public-plus-HGB pipeline. The last registered reader/control jobs are being completed, not used to launch another network search.

## Registered joint-reader result (complete)

The same HGB risk reader, same nested query tasks, and fixed Universal-P definitions give:

| Source → target | Prediction only | Quantity only | Support prior | Old HGB prior | Pointwise prior | DeepSets prior |
|---|---:|---:|---:|---:|---:|---:|
| GAT → Exphormer | 0.742360 | 0.757167 | 0.757966 | 0.784623 | 0.778350 | 0.783894 |
| Exphormer → GAT | 0.749955 | 0.743933 | 0.766981 | 0.772123 | 0.792652 | 0.793870 |

Values are mean seed metrics, equal macro across 20 context/fold strata. All use 1,808 tasks/575 gene clusters in each direction. Differences versus old HGB for pointwise are −0.006273 [−0.028400, 0.029345] and +0.020529 [−0.010745, 0.047469]. Neither satisfies the fixed replacement gate. This is sufficient to stop further neural architecture expansion, while keeping the incumbent as an incumbent rather than claiming it is significantly best.

The McFaline simple same-context Public rule is now frozen for an affected-feedback check against the preserved guide-estimand version. All target-only inputs/scores, feedback cluster order, and label budgets stay fixed. The old Shared score is explicitly a legacy control; it does not become a confirmation of the new Public configuration. No target holdout result selects algorithms or parameters.

The physical-content null is still completing frozen inference and must finish before a mechanism claim about biological content is made. Manuscript/PDF work remains deferred.
