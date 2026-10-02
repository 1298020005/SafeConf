# Registered existing-validation SEEN risk controls

Status: proposal only; no actual numerical features/labels or fits have been read/run. Root approval and an exact independent PASS are required.

The fixed training pool is all2790 eligible validation tasks /1661genes. Arms are TargetOnly(P6), PublicTarget(P6+the frozen LearnedPUBLIC7) and SharedTarget(add original frozen Learned_hgb score). Unsupported2566validation rows receive NaN in all7Public columns and the shared score in a new derived copy. Original features/scores are unchanged; there is no fabricated shared fallback. Existing training-only median/missing flags handle these inputs.

Five nested gene budgets10/25/50/75/100 use SHA256 of `SafeConf_Orion_validation_reuse_20261002_v1|canonical_Ensembl`, retaining every available context for selected genes. Existing rank_labels/CDF helpers see only that budget's validation errors, grouped by dataset/output/upstream/exactcontext-model version/context. Existing fit_risk uses Ridgealpha10 or HGB200/.05/depth3/leaf20/L2=10, seed20260930 and its original gene weights. Exactly30final target-risk fits; no OOF calibration or search.

All models, persisted transforms, budget IDs/CDFs/masks and232-task float64 predictions are hashsealed before any opening of the cached TEST error file. Evaluation is limited to the original232tasks/144genes, two fixed contexts and equal-context macro. All30new models and the three fixed anchors (LearnedSourceHGB, LearnedWeightedDistance, NegativeSourceSupport) remain visible. One shared5000gene-block draw set, seed20260929, keeps original IDs/ties and uses unchanged dec8 metric_values. Only80prespecified contrasts are evaluated; undefined metrics stayNA with valid-draw counts. No plots are generated.

This is a separate retrospective SEEN supplement. Original13primary results, the c94 operation, Source7models/bank and competence-only C records retain their hashes and historical meanings. The2790labels were already exposed for competence; budget entries count downstream uses rather than claiming newly revealed or free zero-label feedback. No Source/Public refit, new upstream call/candidate, raw expression, retrieval or vector load is performed.

Limits:30min wall,2GiB peak RSS,fourCPUthreads,zeroGPU/downloads. Exact code and input hashes are bound in PROPOSED_CONTROL_SCOPE.json and checked before/after. TEST error binding is copied from its completed existing receipt without opening the error file during preparation or fitting.

Root approval schema is `safeconf_orion_existing_validation_risk_control_v1_root_approval`, status APPROVED, with both authorization flags true, exact proposal binding and readonly independent PASS receipt containing that same proposal binding.

```bash
python tools/scripts/run_safeconf_orion_existing_validation_risk_control_agent.py run --root-approval /absolute/path/to/ROOT_APPROVED_CONTROL_SCOPE.json
```
