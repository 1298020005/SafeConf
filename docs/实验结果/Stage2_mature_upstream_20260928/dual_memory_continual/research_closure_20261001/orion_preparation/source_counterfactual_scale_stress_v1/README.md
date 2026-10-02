# Source counterfactual scale stress registration

Status: proposed; actual stress has not run. Root signature and the independent code/proposal PASS are required before execution.

Fixed conditions are s=1, 0.1, 0.05 on the original TxPert GAT and Exphormer DELTA vectors, across K562, RPE1, hepg2 and jurkat. Only the original Manual HGB and prediction-only HGB risk models perform inference. The original cached Manual OOF prior remains fixed. Direct prior RMSE, weighted history distance and magnitude are the three fixed comparators.

This is an in-sample Source DEV/SEEN diagnostic: both final risk models were trained on these same 3616 records. Public OOF priors do not make final risk predictions held out. The scale manipulation is an artificial counterfactual, with no Source mean variant, third upstream candidate, new learner, CDF fit, winner selection or new scientific confirmation.

The runner reproduces all s=1 features and both cached full-batch fitted risk predictions exactly. It writes and freezes all 12 model-scale-upstream conditions before loading Source truth numerically. Later evaluation reports all 120 context-scale-method rows and 30 equal-context macro rows, plus rank/tie and Source error-order diagnostics. No bootstrap or target inputs are accepted. All original Source artifact and code hashes are checked before and after.

Limits: 300 seconds, 1 GiB peak RSS, four CPU threads, zero GPU and downloads. Source truth file bytes may be streamed for registered SHA integrity checks before the seal; numeric truth values are loaded only after the complete prediction seal.

The actual root approval must use schema `safeconf_source_counterfactual_scale_stress_v1_root_approval`, status `APPROVED`, authorization true, the exact proposal binding, and a readonly independent PASS receipt that binds the same proposal.

```bash
python tools/scripts/run_safeconf_source_counterfactual_scale_stress_agent.py run --root-approval /absolute/path/to/ROOT_APPROVED_SCOPE.json
```

Large record/vector artifacts remain in the new isolated runtime output. Existing Source, Orion and serving artifacts are unchanged.
