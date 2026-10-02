# Related work: closest methodological comparisons

This file supports Section 2 of the manuscript. Entries are primary papers or the authors' software repository. It is a focused comparison, not a claim that an exhaustive novelty search has found no predecessor.

| Work | What is already established | SafeConf comparison and action |
|---|---|---|
| [Risk Advisor, Machine Learning 2023](https://link.springer.com/article/10.1007/s10994-022-06248-y) | Model-agnostic post-hoc failure estimation for trained classifiers | Do not claim invention of auxiliary risk learning. Define vector-effect task risk and distinguish source predictor supervision from target feedback. |
| [PertEMA, author software](https://github.com/OfficialBishal/PertEMA) | Perturbation reliability using OOF errors, gradient boosting, calibration and uncertainty intervals; cross-screen refitting is required by its documentation | Treat as an adapted comparator and cite software. The current same-information and native-control adaptations do not reproduce every original uncertainty guarantee. |
| [PRESCRIBE, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/d6383e7643415842b48a5077a1b09c98-Paper-Conference.pdf) | A dedicated perturbation model combines local training evidence and distributional uncertainty | Similarity and quality are established ideas. Compare interface, supervision and data requirements; do not equate its native architecture with a generic frozen-predictor wrapper. |
| [Wang et al., 2026 preprint](https://www.biorxiv.org/content/10.64898/2026.08.11.744177v1) | Experimental reproducibility, shared/specific responses, and quality filtering can change measured model performance | Quality-only and support-only explanations belong in the evidence chain. Quality classification itself is not the contribution. |
| [Nicol et al., 2026 preprint](https://www.biorxiv.org/content/10.64898/2026.05.07.723486v1) | Shared controls can inflate correlation and cosine evaluations of differential responses | Preserve control provenance. Where raw DEV replicates permit, use independent-control checks. Do not generalize the paper's finding into an unsupported statement that every RMSE comparison is invalid. |
| [Deep Sets, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) | Permutation-invariant set representations | Cite the operator. Test whether learned history interactions add value over pointwise scoring and existing handcrafted summaries. |
| [Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html) | Attention-based interactions among set elements | Architecture alternative, currently not an additional experimental arm. More expressive interaction is not a presumed benefit for one-to-three-record sets. |
| [PerturBench](https://arxiv.org/abs/2408.10609) and [author repository](https://github.com/schmons/perturbench) | Unified perturbation model/data evaluation infrastructure | Cite the source of the external pipeline and retain exact local split and data-version provenance; repository dataset labels alone are not sufficient version identification. |

## Candidate contribution and limits

The manuscript investigates a reusable biological reference as an input to error ranking, with source-predictor errors and target feedback controlled as separate information sources. This can be scientifically meaningful without a newly invented general neural layer. A convincing result must demonstrate value beyond a simple reference discrepancy and support, and identify when source error relationships transfer. Cross-architecture results provide one part of this chain; they do not establish cross-family or cross-study performance by themselves.

The PublicSet extension asks a narrower question than whether neural networks are stronger than trees: can a query-conditioned representation of the whole eligible historical set select a more useful biological reference than the same inputs processed per record? Its adoption requires actual reconstruction and downstream risk results. The manuscript currently reports no PublicSet improvement.

Target personalization follows an established practice of learning from observed errors. SafeConf can use such a learner as a component. The experimental question is whether public features or a source-trained score improve it at the same target-error budget, including labels used by CDF construction and calibration. The existing results support the usefulness of public features at adequate budgets; additional benefit from the shared score is not yet consistent.

## Citation status checked for this draft

- PertEMA: software; no accompanying paper according to the author README. Pinned estimator commit is recorded in the execution config and manuscript.
- Reliable perturbations and shared-control bias: explicitly preprints, not treated as peer-reviewed journal results.
- Risk Advisor, PRESCRIBE, Deep Sets and Set Transformer: use the primary publication links above.
- Formal names, versions and primary citations of the local TxPert/Orion assets must accompany the final released adapter manifest. Local cohort names are not presented as new datasets.
