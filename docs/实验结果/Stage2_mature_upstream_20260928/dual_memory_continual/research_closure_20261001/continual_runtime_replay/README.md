# Isolated real Source DEV continual runtime replay

This replay completed actual Public/Error ingestion, immutable revisions, model resolution, a fixed DEV release gate, and administrative rollback/reload. It uses existing Source DEV/SEEN data only. It opens no Orion or McFaline inputs, changes no frozen files, and trains no upstream predictor.

## What was executed

Public Memory starts with 1,047 Replogle 2022 K562/RPE1 records and appends 961 Nadig 2025 HepG2/Jurkat records. The new study triggers the existing update rule. Existing `PublicMemoryStore.create` and `append` calls are recorded; immutable versions and verified `CURRENT` select the resulting 2,008-record bank.

The existing Manual cell-count-weighted prior and all seven Public features are rebuilt from each version's actual historical vectors. The full 2,840-gene projection is verified against the original native bank. Query-specific upstream-source-training eligibility excludes the current task and all current-context replicates. Known other-context same-perturbation history is allowed; it may share the risk gene fold. This is disclosed separately from risk supervision. No finer guide/plate-disjoint claim is made.

The fixed comparison cohort contains 1,699 biological tasks, 3,398 existing predictor/task records and 563 gene clusters. The 109 tasks without initial-version history are excluded by metadata before fits. Folds 2–3 supply initial training, fold 4 supplies added records, fold 0 is the old-task anchor, and fold 1 is the new-task gate. The initial and added Error batches retain existing `e201_official_frozen` and `e205_official_frozen` composite OOF identifiers, rather than claiming individual checkpoints. GAT and Exphormer are two predictors/architectures within one TxPert family. The immutable producer ledger's `upstream_families=2` label is an accounting error; [derived correction](SOURCE_INFORMATION_ACCOUNTING_CORRECTION.json) preserves and binds it, recording predictors=2 and families=1. The output contract is separately bound to common 2,840 genes.

The Public bank's Replogle-to-Nadig staging is an ingestion sequence. Initial Source-error folds 2–3 already include both studies, so this is not chronological Source-error transfer from Replogle to Nadig.

Error records are really appended and saved as immutable revisions per upstream/version. Updated fitting reads their persisted realised errors. Initial shared-risk annotations use two-fold OOF within folds 2–3; added fold-4 annotations use the frozen initial model. CDFs and preprocessing use only the allowed training records, excluding both anchor and new-task gate labels. The same fixed HGB is used throughout: 200 iterations, learning rate 0.05, depth 3, leaf size 20, L2 10 and seed 20260930.

| Risk version | Training folds | Predictor/task error records | Biological tasks | Gene clusters | Weight sum |
|---|---|---:|---:|---:|---:|
| Initial | 2–3 | 1,372 | 686 | 225 | 1,372 |
| Updated candidate | 2–4 | 2,038 | 1,019 | 336 | 2,038 |

The successful run used four small HGB fits: two initial annotation OOF fits and two versioned risk-core fits. It completed in 43.49 seconds with one numerical thread. The separate failed technical run is preserved and disclosed in [COST_LEDGER.csv](COST_LEDGER.csv).

## Actual quality decision

The updated candidate was **rejected**. Models, parameters, features and predictions were frozen before the DEV gate. All eight context-by-predictor strata per role were finite and had at least 20 tasks. Nonfinite predictions, nonfinite metrics, duplicate/missing strata and identity mismatches fail closed. The four role-by-version macro rows are provided in [APPEND_READY_ROLE_VERSION_MACRO.csv](APPEND_READY_ROLE_VERSION_MACRO.csv).

| Gate quantity | Observed | Fixed requirement | Result |
|---|---:|---:|---|
| New-task macro ΔU20 | −0.017995 | ≥ −0.005 | Fail |
| Old-anchor relative AURC change | −0.012739 | ≤ 0.05 | Pass |
| Maximum old/new macro miss-rate increase | 0.016544 | ≤ 0.02 | Pass |
| New-task nonnegative U20 strata | 3/8 = 0.375 | ≥ 0.60 | Fail |

The initial `risk-v1` remains serving with its pinned Public-v1 features. Ingested Public v2 and both updated Error banks remain stored. The rejected model was never published, and no second candidate, parameter search or fit followed the gate. [Actual decision](actual_v2/ACTUAL_RELEASE_OR_RETAIN_DECISION.json) and [old/new task metrics](actual_v2/REAL_DEV_ANCHOR_NEW_TASK_METRICS.csv) retain the full evidence.

The Manual biological prior's mean RMSE decreases in each of the four contexts for both DEV roles. This retrieval result does not imply an improved risk model: new-task U20 decreased. Public records and Source-error records changed together, so this replay cannot causally assign the risk change to either update.

## Rollback and serving evidence

The actual model registry and serving pointer resolve immutable artifact copies and verify artifact, split and schema hashes. Because the candidate failed, the administrative drill first published a version marker referencing the **identical approved v1 artifact**. It then invoked actual registry rollback and restored the v1 serving pointer. This is a deterministic administrative drill, not a real rollback from a quality-approved v2 or an observed performance failure.

The Public pointer was separately rolled from v2 to immutable v1. Rebuilt v1 priors, v1 risk predictions and reloaded model predictions were byte-identical to their originals. The ingestion pointer was then restored to v2 while serving remained on the gate-selected v1 configuration. Added records were not deleted. [Rollback proof](actual_v2/ACTUAL_ROLLBACK_RELOAD_PROOF.json) records the artifact hashes and comparisons.

## Scope of the result

Proven here: real Public retrieval rebuilding, real Error ingestion, shared Source risk-core refitting, immutable versions, fail-closed quality rejection/retention, serving resolution and administrative pointer/artifact restoration.

Not evidenced by this replay: retraining a Public Biology Learner, model-specific residual-adapter updating, a new external C-feedback gain, universal or monotonic improvement, independent new biological observations, or a prospective external confirmation. These are retrospective operations on already available real Source data. Existing Public-growth and C-feedback benefit studies remain separate evidence.

## Preserved technical failure and correction

The first isolated run stopped after publishing its initial model and creating Public v2. Six reconstructed features matched the original at roundoff; cosine differed by up to 1.55e-6. Source prediction files are F-order float32 arrays; selecting rows first makes a C-order copy and changes float32 norm reduction order. Using the unchanged original cosine formula on the full prediction array, then selecting rows, restores the original cosine within 4.44e-16. Neither formula nor tolerance was changed. [Diagnosis and mismatch table](failure_diagnosis_v1/DIAGNOSIS.json) bind the preserved failed status; the retry uses a new runtime/report directory.

## Implementation and review

The new `tools/safeconf_continual/versioned_runtime.py` wraps the existing memory classes without changing them. The new replay entry is `tools/scripts/run_safeconf_continual_runtime_replay_agent.py`. The technical wrapper passed 43 generated checks and the final replay integration passed 33 independent generated checks. Real data were used only by the authorized actual replay. [Final review](wrapper_fixture/FINAL_REPLAY_SCRIPT_REVIEW.json) binds final code hashes.

Completed or partial replay roots cannot be overwritten. A reproduction must specify fresh `--runtime-root` and `--report-root` paths and retain the same registered masks, provenance, features and parameters. [OWNED_FILES_FINAL.json](OWNED_FILES_FINAL.json) lists the small reviewable files; models, feature matrices, priors and event logs remain in isolated server runtime storage.
