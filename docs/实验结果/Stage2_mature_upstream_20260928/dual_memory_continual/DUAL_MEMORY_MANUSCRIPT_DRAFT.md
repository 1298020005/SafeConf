# SafeConf: Continual Public-Experiment and Model-Error Memory for Post-hoc Reliability Auditing

## Abstract (provisional)

Single-cell perturbation predictors return high-dimensional response vectors,
but they do not indicate which predictions should be trusted or experimentally
reviewed first. We present SafeConf, a post-hoc reliability system for frozen
perturbation predictors. SafeConf separates two evidence channels: a shared
Public Memory of real experimental responses and an exact-version Error Memory
constructed from legally held-out feedback for one upstream model. A shared
risk core combines universal prediction evidence with a quality-audited
biological prior; an optional residual adapter incorporates model-specific
feedback with sample-dependent shrinkage, version isolation, release gates and
rollback. On 1,808 TxPert tasks, a preregistered support-regularised public
memory improves the cold-start Utility@20 point estimate for both TxPert
Exphormer (0.7705 to 0.8032) and GAT (0.7742 to 0.7976). A 5,000-draw
cluster bootstrap is reported for every comparison. The same feedback adapter
does not produce a stable Utility@20 gain over the shared core, which defines a
useful negative boundary rather than being hidden. An independent McFaline
study and two published upstream candidates are being evaluated under a
sealed-test contract; those results will determine the external confirmation
route.

## 1. Introduction

Perturbation prediction is usually evaluated by vector reconstruction error.
In deployment, however, the immediate decision is often selective: which
predictions should receive scarce experimental review first? A useful auditor
must operate after a black-box upstream model has produced its prediction, must
not require the target experiment's truth, and should improve as independent
experimental evidence and model-specific feedback accumulate.

Existing reliability approaches primarily exploit the error history of one
predictor. SafeConf treats this as only one source of evidence. Real biological
experiments form a second, model-independent memory that can be shared by new
predictors at cold start. The system therefore separates biological transfer
from predictor-specific error adaptation instead of merging their labels.

Our contributions are:

1. a PredictionRecord and Prediction/Error Contract that maps direct-effect and
   treated-state predictors onto a common, leakage-audited effect space;
2. a versioned Public Memory Bank with explicit alignment, eligibility,
   support, quality and conflict fields;
3. a Shared Risk Core that uses a common error-rank scale across upstream
   architectures and keeps identical biological tasks in the same fold;
4. an exact-version Error Memory adapter with feedback-dependent shrinkage,
   release gates and rollback; and
5. a complete diagnostic report that retains the current negative feedback
   boundary instead of selecting only favorable budgets.

## 2. Problem formulation and contracts

For task $t$, an upstream predictor emits an effect vector $\hat y_t$ under a
fixed `output_contract_id`. The aligned experimental response is $y_t$ and the
task-level error is RMSE on the common gene intersection. The auditor observes
the prediction, task metadata available before truth, Public Memory evidence,
and, when deployed long enough, legal OOF/held-out errors for the same frozen
upstream version. It must output a risk score whose ordering prioritises high
error tasks.

The adapter supports two legal paths. A direct-effect model supplies
`predicted_effect` directly. A treated-state model supplies a treated state and
we subtract a matched predicted or observed control. The conversion rule,
control source, normalization, gene mapping and missing-gene mask are stored in
the PredictionRecord. Incompatible effect contracts are never pooled by raw
RMSE.

## 3. SafeConf architecture

### 3.1 Public Memory Bank

Each memory item contains an experimental response, study and context identity,
perturbation and condition, effect contract, replicate/guide/plate/batch
metadata, provenance and an eligibility record. The bank distinguishes:

- **Support:** cells, guides, plates, batches and independent sources;
- **Quality:** guide/plate reproducibility, replicate consistency, split-half
  stability and batch agreement;
- **Relevance:** context, target and condition similarity to a query task;
- **Conflict:** disagreement across eligible sources; and
- **Content:** the aligned biological effect vector and its uncertainty.

Support is not used as a proxy for quality. For an outer-test task, the bank
excludes the task itself, its replicate group, near-duplicate batches and any
statistic derived from its truth. Public history is therefore prediction-time
information rather than a disguised target label.

### 3.2 Public Biological Prior

For each query task, eligible history is retrieved using only outer-train
statistics. A transfer learner estimates which sources are useful for
reconstructing an unseen biological effect. The current TxPert implementation
uses a shallow HGB transfer score and a fixed, pre-registered 50:50 blend with
cell-count support. This single repair was registered before rerunning the
downstream risk comparison because the unregularised learner over-concentrated
on one source.

The prior exposes an expected effect, uncertainty, effective support and source
conflict. It never receives an upstream error label.

### 3.3 Shared Risk Core

Universal prediction evidence includes magnitude, absolute and signed means,
dispersion, upper quantile and sparsity/shape. The core additionally receives
prediction-prior discrepancy, prior uncertainty, support and conflict. It is
trained on a frozen [0,1] error-rank scale within each outer fold. Dataset and
upstream identifiers, as well as optional-field missingness that could identify
an upstream family, are excluded.

All predictions for one exact biological task share a fold across upstream
models. Outer folds are grouped by perturbation cluster, so an architecture
cannot place the same truth in training while another architecture places it in
test.

### 3.4 Model Error Memory

For a frozen upstream version, a legal feedback item is formed only after the
corresponding held-out truth is returned. The adapter predicts
$r_t - s_t$, where $r_t$ is the realized error rank and $s_t$ is the shared
risk. Its influence is shrunk as

$$\lambda(n)=\frac{n}{n+\kappa},$$

with $\kappa$ selected only inside development folds. Public and error memory
are persisted separately; changing the upstream checkpoint starts a new error
bank. New memory versions are published only if new-task and anchor-task gates
pass, otherwise the previous version is restored.

## 4. Experiments

### 4.1 TxPert public-memory development

The TxPert bank contains 2,008 physically independent source experiments and
5,238 eligible target-to-source edges for 1,808 prediction tasks. GAT and
Exphormer predictions use identical biological-task folds. We compare uniform,
cell-weighted, nearest-control, learned and repaired learned retrieval, then
feed the resulting prior into prediction-only and public-memory risk cores.

### 4.2 Model-feedback development

GAT and Exphormer each have an exact-version ErrorMemoryRegistry with 1,808
legal OOF items. Feedback budgets are 10%, 25%, 50%, 75% and 100% of
perturbation clusters. Every method uses the same tasks, folds and budget. We
report Shared Core, residual Ridge, residual HGB, shuffled-error and
wrong-upstream controls, and a clearly labelled PertEMA-style proxy. The
official PertEMA package is not claimed to be numerically comparable to TxPert;
its CD4-specific feature contract is audited separately.

### 4.3 External confirmation

McFaline train/validation contains 7,173 quality-labelled experiment units, with
100% guide, plate and deterministic split-half coverage. Two published
PerturBench candidates are trained under a registered validation-only budget.
The candidate is selected from validation competence and provenance before any
SafeConf test result is read. Cold-start and continual-adaptation confirmation
are reported separately. For continual adaptation, all feedback-budget
predictions are generated before the permanent holdout truth is opened.

## 5. Results available before external confirmation

### 5.1 Public biology and cold-start risk

The repaired transfer learner reduces effect reconstruction RMSE from 0.062924
to 0.062113 and increases cosine from 0.509135 to 0.529424 relative to the
unregularised learned retrieval. Downstream Utility@20 is:

| upstream | prediction-only HGB | manual public HGB | repaired learned public HGB |
|---|---:|---:|---:|
| TxPert Exphormer | 0.770472 | 0.789953 | **0.803162** |
| TxPert GAT | 0.774242 | 0.788090 | **0.797582** |

The repaired-minus-manual paired cluster bootstrap intervals include zero
([−0.0087, 0.0302] for Exphormer and [−0.0105, 0.0325] for GAT). The permitted
claim is therefore a positive, cross-architecture point-estimate direction
with limited precision, not universal significance.

### 5.2 Error-memory boundary

Across all five feedback budgets, residual HGB has lower feedback-curve U20
area than the frozen Shared Core for both architectures (Exphormer 0.790206
versus 0.803162; GAT 0.792046 versus 0.797582). The paired bootstrap intervals
include zero at individual budgets, but the point estimates do not support a
primary universal gain claim. The result is retained as a model-specific
feedback boundary and as evidence that public biological memory and model-error
memory should not be conflated.

## 6. Discussion and current decision

The current evidence favors Public Memory + Shared Risk Core as the main method
candidate. The Error Adapter is implemented, isolated, and auditable, but is
not promoted to the main performance contribution until an external feedback
curve is non-inferior. If the McFaline cold-start confirmation supports the
shared core, the paper follows the continual Public-Memory route and presents
Error Memory as an optional deployment adaptation with its negative boundary.
If external confirmation fails, the failure is reported together with the
registered diagnosis and the paper is narrowed to the reliability boundary
that is reproducible across the available architectures.

## 7. Reproducibility

The source code, memory schemas, update manager, exact-version registries,
bootstrap outputs, result tables and figure are committed under the
`dual_memory_continual` experiment directory. The external test partition
remains sealed until the validation competence gate and the complete cold-start
configuration are frozen.
