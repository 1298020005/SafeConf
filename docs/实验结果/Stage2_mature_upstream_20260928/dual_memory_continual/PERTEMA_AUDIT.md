# PertEMA comparison audit

- Official repository: `https://github.com/OfficialBishal/PertEMA`
- Audited commit: `43c09a32e23d0ee2ae5dfbab21b2deeab27f1803`
- License: MIT (repository metadata).
- Official estimator: XGBoost meta-model with isotonic calibration and split-conformal interval.
- Official feature contract: 64 prediction-time features, including CD4 activation contexts (`Rest`, `Stim8hr`, `Stim48hr`) and a bundled reference embedding.
- Official README explicitly states that the shipped estimator is trained on one CD4 screen and should be refit for a new screen.

## Consequence for SafeConf

The frozen official CD4 estimator cannot be applied as a numerically fair comparator to TxPert E201/E205: the context and reference feature contracts are different, and the repository does not provide a TxPert adapter. We therefore report a **PertEMA-style proxy** only: a shallow HGB error learner using the same TxPert OOF feedback clusters and the same prediction-only feature budget. It is labelled `PertEMA_style_HGB_P_proxy` in all new result files, never `PertEMA`.

An official PertEMA run becomes a formal comparator only after an adapter is trained and frozen on the same upstream/output contract with identical OOF error budgets. No current result claims that this compatibility step has been completed.
