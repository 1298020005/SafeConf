# A1 task-versus-predictor audit

Source: `/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/safeconf_reliability_model_corrected_20260610/tables/RELIABILITY_ALL_SCORES.csv`

## One-line decision

Task difficulty is the leading narrative.

## Overall summary

- Paired test tasks: 4584
- Spearman(V0 error, ContextSim error): 0.973
- Pearson(V0 error, ContextSim error): 0.884
- Task variance fraction: 0.936
- Predictor variance fraction: 0.001
- Paired Cohen's d: 0.130

## Dataset-level count

- Task-difficulty dominant: 7/7
- Predictor-difference nontrivial: 0/7

## Interpretation rule

- If task variance dominates in most datasets and predictor errors are strongly correlated, SafeConf should be framed as predictor-agnostic task-risk scoring.
- If predictor variance or paired Cohen's d is large in several datasets, start A2 as a minimal model-aware extension.
