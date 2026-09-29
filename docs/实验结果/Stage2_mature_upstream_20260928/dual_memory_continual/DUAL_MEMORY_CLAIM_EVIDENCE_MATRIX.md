# Dual-Memory claim–evidence matrix

| Claim | Evidence | Current status | Permitted wording |
|---|---|---|---|
| Public memory is model-independent | TxPert bank stores biological effects and no upstream error; same bank used by GAT/Exphormer | supported | shared public experimental memory |
| Public-memory benefit is cross-architecture | identical biological tasks and folds for GAT/Exphormer; repaired public HGB improves both | supported on one family | positive within-family cross-architecture direction |
| Risk core transfers to an unseen predictor | source-upstream error labels only; target-upstream errors and the same biological fold excluded | positive in both directions; GAT→Exphormer CI above zero, reverse CI crosses zero | cross-predictor transfer with asymmetric statistical precision |
| Learned public retrieval is useful | fixed 50:50 support regularisation improves RMSE/cosine and U20 point estimates | positive but CI-limited | preregistered repair improves the observed DEV/SEEN direction |
| Quality is available in a real study | McFaline train/val: 7,173 units, 100% guide/plate/split-half coverage | asset validated, outcome pending | quality-labelled public memory asset |
| Error Memory is model-specific | separate exact-version banks, wrong-upstream and shuffled controls | pipeline supported; repaired run measured | model-version-specific feedback adapter |
| Error feedback always improves | repaired residual HGB is below Shared Core at most budgets on both architectures | not supported | optional feedback adaptation and explicit negative boundary; no universal gain claim |
| Official PertEMA is directly comparable | official model is CD4-specific with a different 64-feature contract | false for current TxPert setup | PertEMA-style proxy only until same-contract refit |
| External upstream competence | LatentAdditive failed; raw DecoderOnly narrowly failed; one preregistered validation-only OOF shrinkage repair beat the strongest baseline by 0.8734% with all 3 strata non-inferior | supported before SafeConf/test access | validation-calibrated DecoderOnly is the sole frozen external upstream |
| External SafeConf generalisation | McFaline test remains sealed; no SafeConf result was used to select the external upstream | pending | external cold-start confirmation pending |
