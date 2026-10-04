# SafeConf implementation failure and action log

- Source gate first run failed in summary index handling; aggregation was fixed and the complete real plus five shuffled-label pipelines were rerun in `source_gate_v3`.
- Native61 contains non-finite `native_training_similarity`; the current-truth adaptation preserves task identity coverage and uses the fixed XGBoost missing-value path. It is reported separately from P6.
- PublicMeanJackknife is complete for 210/212 holdout tasks. Missing values are not imputed; the system falls back to PublicRule and records the coverage failure.
- Target TabPFN passed the DEV technical gate but lost on the frozen current holdout. It remains a capacity audit and does not replace H1/XGB or PublicRule.
- Current-truth PertEMA P6 and Native61 are reproducible adaptations under the current truth contract; the result is not labelled as a complete official conformal PertEMA reproduction.
- Mixed history/no-history ranking was completed from the pre-sealed 2,993-task score artifact. It is a SEEN retrospective audit, not independent confirmation; no permanent TEST truth was opened, no new download was made, and no E208 process was changed.
