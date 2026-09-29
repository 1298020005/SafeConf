# Dual-Memory SafeConf manuscript outline

## Working title

**SafeConf: Public-Experiment Memory for Cross-Predictor Reliability Auditing
of Single-Cell Perturbation Predictions**

## 1. Problem

Existing perturbation predictors return a vector but not a trustworthy way to
prioritise experimental review. A post-hoc auditor should work with a frozen
black-box predictor, use real biological evidence available before the target
truth, and improve as public experiments and model-specific feedback arrive.

## 2. System

1. `PredictionRecord` and Prediction/Error Contract.
2. Public Memory Bank: alignment, eligibility, support, quality, conflict and
   biological transfer.
3. Shared Risk Core: universal prediction evidence plus the public biological
   prior, trained on a common `[0,1]` error-rank scale.
4. Model Error Memory: exact upstream-version OOF/held-out errors and a
   residual adapter with feedback-dependent shrinkage.
5. Versioned updates, release gates, rollback and no-forgetting anchors.

## 3. Experiments

1. Biological reconstruction: uniform, support-weighted, nearest, learned and
   preregistered support-regularised retrieval.
2. Shared cold-start risk: magnitude, prediction-only, manual public prior and
   repaired learned public prior on TxPert GAT/Exphormer.
3. Cross-architecture transfer with the same biological task in one fold.
   The primary version is zero-target-label transfer in both directions; it
   excludes the queried biological fold from source-upstream training.
4. Error Memory learning curves at 10/25/50/75/100% cluster feedback,
   including shuffled and wrong-upstream controls.
5. McFaline upstream competence, then sealed cold-start and continual
   adaptation confirmation.

The current figure draft is
`figures/DUAL_MEMORY_PUBLIC_ERROR_RESULTS.svg`: it keeps the public-memory
gain and the error-memory boundary in one paired view.

## 4. Results claims permitted by current evidence

- Public biological memory provides a positive cross-architecture signal on
  TxPert DEV/SEEN.
- A fixed support regularisation repair prevents learned retrieval collapse and
  improves both effect reconstruction and downstream risk point estimates.
- A source-upstream risk learner transfers without target-predictor error
  labels in both directions; GAT-to-Exphormer has a positive paired-bootstrap
  lower bound and the reverse direction remains positive but uncertain.
- Error Memory is a valid model-version-isolated update path; its U20 benefit
  is not yet stable enough to be the primary contribution.
- McFaline has a real quality-labelled train/validation public bank, while its
  test truth remains sealed.

## 5. Discussion and routes

- Route A if external cold-start and feedback confirmation both pass.
- Route B if public-memory/shared-core confirmation passes but Error Adapter is
  only useful after sufficient feedback.
- Route C only if a second, independent failure boundary is reproducible.

No route may claim broad model-agnostic generalisation from TxPert alone.
