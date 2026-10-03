# v0.4.2 diagnosis and actions

## Completed

- The rule and supervised candidates now share one `score_task(task, frozen_config)` dispatch interface without forcing the original public rule through a label-trained model.
- Source Ridge uses only E201 source error labels and fold-out-of-fold Shared scores. Target errors are not read during its fit.
- Target Ridge uses the registered McFaline development labels; its 212-task holdout is evaluation-only.
- The complete current-holdout ranking contains the same 212 tasks for every required candidate, with stable task-ID tie breaking.
- Channel-score CDFs and error-label CDFs are written as separate audits.

## Decisions

- Keep the original PublicRule as the current default. It is the strongest current same-truth no-error-label candidate.
- Keep SourceRidge and TargetRidge50 as conditional candidates. Their point gains do not pass the paired bootstrap adoption gate on the current contract.
- Keep the legacy PertEMA comparison as an excluded audit because its stored truth contract differs from the current holdout; do not describe it as a fair current score.

## Continuing action

SAMS training remains protected and active. Once it reaches a terminal checkpoint, run the registered truth-free generation, competence screen and bidirectional transfer protocol. No new network or Ridge search is opened by this audit.
