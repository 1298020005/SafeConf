# Dual-Memory SafeConf: current method decision

## Decision state

The implementation now contains the persistent public and model-specific
memory contracts. The method is still in DEV/SEEN evaluation until McFaline
validation competence and the sealed external test are complete.

## What the first real experiments establish

### Public Memory and Shared Risk Core

TxPert E201/E205 uses 1,808 biological tasks, 2,008 physically distinct
public-memory experiments and 5,238 eligibility edges. GAT and Exphormer
predictions for an identical biological task share the same gene-cluster outer
fold. The public learner never receives an upstream error label.

| upstream | prediction-only HGB U20 | manual public HGB U20 | repaired learned public HGB U20 |
|---|---:|---:|---:|
| TxPert Exphormer | 0.770472 | 0.789953 | **0.803162** |
| TxPert GAT | 0.774242 | 0.788090 | **0.797582** |

The repaired learner uses a preregistered fixed 50:50 blend of learned
transfer weights and cell-count support weights. It improved biological
effect reconstruction to RMSE `0.062113`, cosine `0.529424`, and effective
source count `2.118`. The unrepaired learner had higher, worse RMSE (`0.062924`) and
over-concentrated retrieval (effective sources `1.667`) and lower cosine
(`0.509135`). No second retrieval repair is allowed.

The paired bootstrap intervals for the repaired-vs-manual downstream U20
comparison still include zero. The correct claim is therefore a positive,
cross-architecture direction with limited statistical precision, not a claim
of universal significance.

### Zero-target-label cross-predictor transfer

A dedicated transfer experiment now fits every preprocessing statistic, error
CDF and risk learner on one source upstream only. The target upstream provides
zero error labels, and source records from the target biological fold are
excluded. Repaired-public HGB improves over target magnitude in both directions:

| risk-label source | unseen target predictor | magnitude U20 | transferred U20 | delta | 95% cluster CI |
|---|---|---:|---:|---:|---:|
| Exphormer | GAT | 0.768216 | 0.794104 | +0.027936 | [-0.008901, 0.072414] |
| GAT | Exphormer | 0.758726 | 0.795157 | +0.037366 | [0.002665, 0.082318] |

This supports cross-predictor risk transfer, with statistically precise support
in one direction and a positive but uncertain reverse direction. It does not
yet establish transfer across an independent model family or study.

### Error Memory and Error Adapter

The Error Memory is separate for the exact model versions `E201_STRING_GAT`
and `E205_Exphormer`. It contains only legal gene-disjoint OOF errors. The
first run with the repaired Shared Core is recorded in
`error_adaptation_repaired/`.

The feedback experiment uses 10/25/50/75/100% perturbation-cluster budgets,
the same tasks and folds for every method, and 5,000 cluster bootstrap draws.
The model-specific residual adapter is evaluated against the frozen shared
core, a shuffled-error control, a wrong-upstream control, and a
PertEMA-style HGB proxy. The official PertEMA package is not mislabeled as a
TxPert result; its CD4-specific 64-feature contract is recorded in
`PERTEMA_AUDIT.md`.

The repaired run does **not** meet that primary-gain rule: Residual HGB U20 is
below the Shared Core at most budgets for both architectures, while the
model-error proxy is consistently weaker. Error Memory therefore remains a
valid, version-isolated deployment extension and a negative boundary result,
not a primary performance contribution. It is retained because the system can
ingest real feedback, shrink its influence at low sample counts, and apply
release/rollback gates without contaminating the shared public bank.

## Chosen system structure

```text
PublicMemoryStore
  -> PublicBiologyLearner (biological transfer only)
  -> Public Biological Prior
  -> SharedRiskCore (universal prediction evidence + prior discrepancy)
  -> Shared risk rank in [0,1]
  -> exact-version ErrorMemoryRegistry
  -> ErrorResidualAdapter with feedback-dependent shrinkage
  -> final risk rank in [0,1]
```

This is a continual system rather than a learned gate. Public evidence is
shared across upstreams; realised error evidence is isolated by upstream
model and checkpoint version. `ContinualUpdateManager` implements the
pre-registered update triggers and release/rollback gates.

## External confirmation state

McFaline train/validation contains 7,173 quality-labelled experiment units.
Guide, plate and deterministic split-half quality coverage is 100%; test
expression has not been aggregated. Two published PerturBench candidates are
being trained under a registered validation-only epoch budget, with the test
partition still sealed. The final external route is selected by upstream
competence before any SafeConf test result is read.

Until that step completes, the paper claim is:

> A cross-architecture post-hoc reliability system can combine a shared,
> quality-audited public experimental memory with model-version-specific OOF
> error memory; the public-memory contribution is supported on TxPert DEV/SEEN,
> while external generalisation and continual adaptation remain confirmation
> tests.
