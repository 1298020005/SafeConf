---
title: "SafeConf: a fail-closed post-prediction reliability contract for single-cell perturbation models"
format: Nature-style manuscript draft (GLM v1, 2026-08-17)
note: >
  All numbers come from frozen evaluation reports committed in the SafeConf repository.
  E201 (four-context blind audit) results are NOT released yet; every pending value is an
  explicit [[E201-PLACEHOLDER]]. Replace placeholders only after STAGE_7 of
  agents/glm/04_E201_RUNBOOK.md completes. Do not soften or delete negative results.
authors:
  - "Yifan Yang¹"
  - "[co-authors / corresponding author: to be confirmed with supervisor]"
affiliation: "¹ School of Computer and Information Engineering, Henan University, Kaifeng, Henan, China"
target_journals: "Primary: Briefings in Bioinformatics (JCR Q1; CAS minor-category 1). Stretch: Genome Biology. Backup: Bioinformatics (Applications)."
---

# SafeConf: a fail-closed post-prediction reliability contract for single-cell perturbation models

**Yifan Yang¹** &nbsp; [co-authors: to be confirmed]

¹ School of Computer and Information Engineering, Henan University, Kaifeng, Henan, China

## Abstract

Computational models that predict transcriptional responses to genetic and chemical perturbations are increasingly used to plan experiments, yet a deployed prediction arrives with no reliable statement of which tasks a given uncertainty signal may be trusted on. We present SafeConf, a model-agnostic auditing layer that operates strictly *after* predictions have been issued and *before* target measurements exist. SafeConf scores task-level risk from deployment-time signals only — family disagreement across frozen model seeds, predicted effect magnitude, and source-context support — and couples every signal to a pre-registered *validation footprint*: the set of evaluation settings in which the signal has demonstrated validity. Where the footprint does not extend — cross-study transfer, double-unseen tasks, settings where a competitor's confidence score saturates — the contract fails closed and returns ABSTAIN rather than a ranking. Across a public TxPert STRING-GAT retraining audit on K562 (263 unseen-gene tasks and 566 full-context-holdout tasks), the same two signals exchanged roles: disagreement ranked errors where magnitude could not (Spearman ρ = 0.395, 95% CI 0.283–0.497, versus 0.096, CI −0.026–0.219), while magnitude dominated under context holdout (ρ = 0.880 versus 0.424; 20%-budget review utility 0.913 versus 0.365). A preregistered cross-study transfer to RPE1 correctly abstained (ρ = 0.300, CI −0.040–0.580), and the official confidence scores of PRESCRIBE, an evidential competitor, saturated on strictly unseen genes, making rank statistics undefined. [[E201-PLACEHOLDER-ABSTRACT: one sentence on the four-cell-line leave-one-out blind audit (K562/RPE1/HepG2/Jurkat, 4 seeds, 2,008 tasks), to be inserted after the sealed evaluation completes.]] SafeConf thus turns reliability from a property claimed by a predictor into a property audited per setting, with negative results retained.

*(Alternative abstract variant B, for the case where the E201 routing gate fails on most targets, is kept in the project workspace and leads with the failure-boundary framing; the variant is chosen only after the frozen gates are evaluated — never before.)*

## Introduction

Perturb-seq and related assays couple pooled genetic screens with single-cell readouts, and a growing family of models — GEARS, CPA, scGPT, and the knowledge-graph-guided TxPert — predicts transcriptional responses to perturbations that have never been measured [1–5]. Benchmarks have simultaneously shown that the harder the deployment condition, the larger the apparent performance drop: unseen contexts, unseen perturbations, and cross-dataset transfer degrade accuracy markedly, and simple baselines frequently remain competitive [6–9]. A practitioner holding a panel of thousands of predicted effects therefore faces a question none of these systems answers on its own: *which of these predictions deserve scarce manual review or wet-lab validation, and when is no existing signal entitled to rank them at all?*

Uncertainty quantification for perturbation prediction exists, but as a property of particular predictors. GEARS exposes a variance head and has used its internal uncertainty to prioritise combinatorial perturbations for experimental validation [2]. PRESCRIBE embeds evidential regression in the predictor and filters low-confidence outputs [5]. GPerturb represents gene-level effects with Gaussian-process posteriors [10]. Conformal-style coverage for perturbation predictors has recently been explored post hoc (ConfPert, ICML 2026 workshop) [11]. These approaches provide numbers per prediction; they do not provide a *contract* stating, per evaluation setting, whether a given deployment-time signal has demonstrated validity — and none fails closed when it has not. Meanwhile, agents in omics have begun to adopt calibrated abstention (Medea) [12], and general-purpose post-hoc risk estimation over black-box outputs is established in machine learning (Risk Advisor) [13]; neither is tied to perturbation prediction or to hard biological hold-outs. A recent evaluation-methodology study explicitly notes that calibration of uncertainty estimates for perturbation prediction "is rarely evaluated at present" [9].

SafeConf occupies that gap deliberately narrowly. It is not a predictor and adds no new uncertainty theorem; its ingredients are an ensemble-disagreement identity as old as ensemble learning [14], simple deployment-time statistics, and pre-registered decision gates. Its contribution is the *contract*: (i) every risk signal is computed only from information available at deployment time (frozen predictions and metadata, never target measurements); (ii) every signal carries an explicit validation footprint — the settings in which it has passed pre-registered gates — and may be used for ranking only inside that footprint; (iii) outside the footprint the system returns ABSTAIN; and (iv) the entire evidence chain is sealed by hashing and dual-repository commits *before* any target truth is opened, so that post-hoc repair is visible. We show that this discipline is not bureaucratic caution but reflects a real empirical structure: **the validity of a risk signal flips with the evaluation setting**, and a deployment layer that ignores the flip misleads.

Three questions posed by our supervisor organise the evaluation, and we answer each under progressively harder missing-data structures: (1) *whose* error does a risk score address (which predictor, which error definition)? (2) does any signal secretly require the measurement it is supposed to anticipate? and (3) do conclusions survive small training submatrices, whole-row/whole-column hold-outs, and cross-dataset transfer [15]? The central experiment, E201, extends the audit to the setting the community now treats as standard — leave-one-cell-line-out across K562, RPE1, HepG2 and Jurkat [3,16] — under a fully blind, pre-registered protocol: sixteen TxPert STRING-GAT retrainings (four targets × four seeds), 2,008 context–perturbation tasks, with predictions, risk tables and baselines committed to two independent remotes before the first target expression value is read. [[E201-PLACEHOLDER-INTRO: replace this sentence with the one-line outcome after adjudication.]]

## Results

### A fail-closed contract over deployment-time signals

SafeConf assumes a frozen family of prediction vectors for each task and computes task-level risk from five signals available before any target measurement: family disagreement around the family centroid; model–baseline gap; dispersion of source-context perturbation effects; source support (negative log source cells); and support-context deficit (how many of the three source contexts lack the perturbation). Predicted effect magnitude is computed identically but is *excluded* from the risk score and retained as the primary simple baseline, because "bigger predicted change ⇒ harder task" is a strong, hard-to-beat heuristic (Fig. 1a). For a registered family of m equally weighted members p_i with centroid p̄, the squared-error ambiguity identity

  R_F² = ‖p̄ − y‖² + D_F²,  D_F² = (1/m) Σᵢ ‖pᵢ − p̄‖²,

links family RMS error R_F to the observable disagreement D_F. We treat this identity exactly as what it is: a deterministic integrity check that makes any tampering with the sealed prediction release detectable row by row (residual ≤ 10⁻¹⁰ in 64-bit arithmetic), not a scientific discovery. The contract's scientific content lives in three pre-registered, separately adjudicated gates (Fig. 1b): a *certificate gate* (the identity holds and no task shows family RMS below disagreement), a *routing gate* (risk–error pooled Spearman CI lower bound > 0 **and** 20%-budget review-utility CI lower bound > 0), and a *magnitude-increment gate* (partial Spearman controlling magnitude, or paired utility increment over magnitude, with CI lower bound > 0). Failing a gate is an outcome, not an error: negative results are retained without re-tuning seeds, weights, tasks or metrics.

Every signal is then published with its validation footprint (Fig. 1c): the settings in which it currently holds, those in which it is untested, and those in which it has degraded or become undefined. The footprint, not an average score, is the deliverable to a deployment team.

### Signal validity flips with the evaluation setting

Under a public TxPert STRING-GAT retraining audit inside K562 — the easiest of our hard settings, where unseen genes must be predicted in a familiar context — family disagreement carried usable risk information while predicted magnitude did not: across 263 unseen-gene tasks, disagreement's lower bound correlated with family RMS error at ρ = 0.395 (95% CI 0.283–0.497) and delivered a positive 20%-budget review utility of 0.208 (CI 0.103–0.376), whereas magnitude's correlation was 0.096 with a CI crossing zero (−0.026–0.219) and utility indistinguishable from random selection (Fig. 2, Fig. 3a) [E199 tables].

Holding out the *entire* K562 context — training on three other cell lines and transferring — inverted the picture. Across 566 tasks, our transfer-risk signal still correlated with error (ρ = 0.424, CI 0.351–0.495; utility 0.365, CI 0.236–0.481), but predicted magnitude became the single best available signal by a wide margin (ρ = 0.880, CI 0.844–0.909; utility 0.913, CI 0.875–0.952), with source-effect dispersion intermediate (ρ = 0.664) [E200 tables]. Under context shift, tasks predicted to change a lot really are the tasks that go wrong, and nothing in our signal set beat that heuristic at the review budget.

The same discipline applies to our own negative results. In a preregistered cross-study transfer (Adamson→Replogle, RPE1 target, 175 tasks), disagreement showed ρ = 0.300 with CI −0.040–0.580 — a positive utility point estimate (0.696) but a correlation interval crossing zero — and the pre-registered dual gate returned ABSTAIN [E192]. On double-unseen tasks (neither context nor perturbation observed during training), the disagreement–error association itself turned negative (Spearman −0.349 to −0.241 across support levels [E189]), and the 20%-budget review utility of both disagreement (−0.127) and predicted magnitude (−0.080) fell below random expectation [E191]; random missing-cell splits, by contrast, flatter every signal (association 0.368–0.412), confirming that random hold-outs overstate deployability. We do not average away these reversals; we record them as the footprint's content (Fig. 2). To our knowledge this setting-dependence — including the dominance of a one-line baseline precisely where deployment is hardest — has not previously been documented as an explicit operational rule for perturbation-prediction deployment, although benchmarks have reported the underlying performance drops [6–9].

### What the flip is made of: increment, mechanism, and the composite trap

Three analyses on the frozen tables locate the flip precisely (Fig. 6). First, *paired increments*: in E199 the risk signal adds over magnitude (Δρ = 0.299, CI 0.161–0.432) and magnitude adds nothing once disagreement is known (rank-partial ρ = 0.005); in E200 the sign reverses (Δρ = −0.456, CI −0.536 to −0.378; Δutility = −0.548, CI −0.689 to −0.434) while magnitude retains nearly all its information after controlling the risk score (partial ρ = 0.875). The two signals are nearly orthogonal (ρ = 0.25 in E199) and the dominant information axis switches between settings; under context holdout, predicted magnitude behaves as a proxy for extrapolation distance (ρ = 0.66 with source-effect dispersion, which retains only 0.161 after controlling magnitude). Second, *fixed composites cannot recover the loss*: adding any signal to magnitude dilutes it in E200 (utility 0.916 → 0.822–0.890) while helping mildly in E199 (0.208 → 0.249) — a fixed weighting loses in one regime whatever it gains in the other (Fig. 6c). Third, a *conditional-contract simulation* across nine frozen settings (Fig. 7) shows the practical consequence: deploying the best gate-passing signal per setting and abstaining elsewhere (utility floor 0 by construction) matches fixed scores on average (0.23 vs 0.25–0.26) while eliminating their negative tail (−0.13 and −0.08 in double-unseen tasks); abstention is precisely what protects the cross-study RPE1 case, where both signals' point utilities look attractive (0.70–0.73) yet both correlation CIs cross zero. The simulation is illustrative — its footprint labels are in-sample — and E201 is its first prospective test. Component-wise footprints for every signal appear in Fig. 5.

### Competitor confidence degrades where audits matter

PRESCRIBE, the closest predictor-intrinsic confidence method, ships official combined and epistemic confidence scores [5]. On strictly unseen genes (Norman panels P3/P4, 48 tasks), both official scores were *saturated* — effectively a single constant value across the panel — making rank statistics undefined and confidence filtering vacuous; the correct system behaviour there is ABSTAIN, and the correct scientific statement is that predictor-intrinsic confidence did not survive this distribution shift, not that any method "won" (Fig. 2) [E158/E159]. Post-hoc conformal coverage (ConfPert [11]) provides finite-sample coverage intervals over evaluation metrics for eight predictors; coverage, however, degrades silently under distribution shift and does not by itself decide which tasks to review, which is a triage question. SafeConf is thus complementary to, not a replacement for, both families of methods — and must be compared against them only on non-saturated tasks under a shared task contract (planned as a pre-registered follow-up, E202b).

### A four-cell-line blind audit under leave-one-context-out

E201 asks whether any of the K562 conclusions transport to the setting practitioners actually face: an entirely unseen cellular context. Each of K562, RPE1, HepG2 and Jurkat was held out in turn; the public TxPert STRING-GAT architecture was retrained from scratch on the remaining three contexts (only their controls visible for the target), with four random seeds per target — sixteen retrainings, 80 epochs each, 2,008 context–perturbation tasks (1,808 with ≥30 cells for the main analysis) (Fig. 4a,b). The protocol is blind in a verifiable sense: the physical training view zeroes all target perturbed expression (542,007 cells), training logs record zero target-truth rows accessed, the prediction program audits at every batch that `batch.x` is all-zero and runs a 0→1 dummy-input invariance check, and predictions, risk tables and the official general baseline are committed to GitHub and Gitee *before* a separate release program opens target truth. An equivalence check ties the baseline to the public TxPert `MeanBaseline` (maximum absolute residual 2.79×10⁻⁶ over 580 K562 tasks against E200 outputs, tolerance 5×10⁻⁶). Adjudication uses the three gates of Fig. 1b with cluster bootstrap (5,000 draws, resampling perturbation conditions across targets jointly); all four targets are reported whatever the direction of results. [[E201-PLACEHOLDER-RESULTS: replace this subsection's outcome paragraph with per-target gate outcomes, pooled intervals, and the magnitude comparison; report every target including adverse ones.]]

### Governance: the certificate belongs to the registered family

Because disagreement-based quantities reward family enlargement, the contract fixes family composition before evaluation: member identities, seeds, weights, gene panel and aggregation are registered, and hash-sealed. Stress tests with duplicated and synthetic members show that naïve diversity inflation is detectable and that governance rules (architecture/lineage weighting, centroid-shift penalties) contain it [E194]; the same registration is what makes the E201 seal auditable. We report these as tamper-evidence properties of the protocol, consistent with the certificate's role as an integrity device rather than a tight error bound.

## Discussion

SafeConf's premise is that reliability claims for perturbation prediction should be audited the way the predictions themselves are now benchmarked: per setting, under hard hold-outs, with simple baselines admitted. Our results support a concrete and uncomfortable conclusion: **no deployment-time signal we tested is valid everywhere, and the strongest signal in the hardest setting is a one-line baseline** — predicted magnitude under full-context holdout. A deployment layer that ships a single learned risk score without a validation footprint will therefore be confidently wrong in exactly the settings where users need it most. The fail-closed contract is our operational answer: rank only inside the verified footprint; abstain elsewhere; make the abstention itself the visible, auditable behaviour.

Several limits bound the present claims. All evidence is computational, on public CRISPR datasets; no wet-lab validation has been performed (GEARS has demonstrated uncertainty-guided experimental prioritisation for its own model [2] — an obvious next step for a model-agnostic layer, not a claim we make here). The E201 audit uses the *public* STRING-GAT TxPert configuration, not the authors' full proprietary graph stack; statements about TxPert are accordingly statements about the public retraining. Chemical perturbations behave differently from genetic ones (magnitude more often dominant in our chemical audits) and are treated as a boundary, not merged into a unified success. PRESCRIBE comparison is currently limited to its saturated strict-unseen-gene regime; a non-saturated head-to-head under a shared contract remains future work (E202b). Finally, the footprint itself is provisional: E201 extends it to four cell lines, but transport to entirely new assays will require new registrations.

[[E201-PLACEHOLDER-DISCUSSION: one paragraph after adjudication — either "the footprint extends to four contexts, with heterogeneity X" or "the routing gate failed in N of four contexts; the deliverable is the abstention rule", per the frozen three-way branching plan.]]

## Methods

### Problem definition and error objects

For one perturbation task with G genes, member i of a frozen family predicts effect vector p_i ∈ ℝ^G; the experimental effect y is sealed. With the gene-normalised norm ‖v‖ = sqrt((1/G)Σ_g v_g²), task error for member i is ‖p_i − y‖; family RMS error is R_F = sqrt((1/m)Σᵢ‖pᵢ − y‖²); worst-member error is W_F = max_i‖pᵢ − y‖. Every risk statement names its predictor (public TxPert STRING-GAT family; earlier audits: scGPT/GEARS families), its error object (family RMS unless stated), its evaluation space (frozen gene panel, effect scale) and its task unit (context–perturbation centroid).

### Deployment-time signals and the risk score

Per task: family disagreement D_F; family radius; predicted magnitude ‖p̄ − matched control‖; model–baseline gap to the count-weighted source-mean transfer; dispersion of the three source-context effects; negative log source cells; support-context deficit. Signals are z-standardised using means/SDs estimated on the target's main tasks only, and the E201 risk score is the equal-weight mean of five components (family disagreement, model–baseline gap, source-effect dispersion, negative log source cells, support-context deficit); predicted magnitude is deliberately excluded and evaluated as the primary baseline.

### Gates, abstention, and the validation footprint

The routing gate requires pooled Spearman CI lower bound > 0 and 20%-budget review-utility CI lower bound > 0 (utility = normalized excess of selected-set mean error over overall mean error, oracle-normalized). The magnitude-increment gate requires partial Spearman or paired utility increment CI lower bound > 0. The certificate gate requires the ambiguity-identity residual within numerical tolerance (10⁻¹⁰) and zero tasks with R_F < D_F. A signal's footprint records, per setting: VALID, UNTESTED, DEGRADED (CI crosses 0), NEGATIVE, or UNDEFINED (score saturation). Outside VALID, the system returns ABSTAIN.

### Deterministic certificates and calibration (inherited components)

The ambiguity identity above and the diameter bound Δ_F/2 ≤ W_F (Δ_F the family diameter) are classical [14] and used as integrity checks. An independently calibrated split-conformal upper bound U for a reference centroid transfers to the registered family via the observable shift s = ‖p̄ − c‖ as R_F ≤ sqrt((U+s)² + D_F²); target-cluster conformal calibration treats repeated guides/contexts within a target as one calibration unit [17]. These components follow the earlier registered-family formulation [18] and are retained for continuity of the evidence chain.

### TxPert public retraining and the E201 blind protocol

Public TxPert code (fixed commit; STRING-GAT configuration; batch 64; 80 epochs; seeds 1–4) is retrained per leave-one-context-out target with a physically blinded H5AD in which all target perturbed expression is zeroed (542,007 cells) and `uns` emptied; source validation alone guides training. Task base: context–perturbation units (2,008 tasks; 1,808 main with ≥30 cells; 200 sensitivity tasks with 10–29 cells), built from the blind view before any truth access. Prediction, risk-feature, baseline and truth-release programs are separate executables with hash re-verification at every start; the release program recomputes all hashes before opening official target expression. Full frozen conditions: TARGET_RELEASE_AND_EVALUATION_FREEZE.md (2026-08-02) in the project repository.

### Datasets and earlier audits

E189 (small training submatrices, random/row/column/double-unseen; 13,440 task instances), E190 (Adamson→Replogle K562 cross-study; 692 tasks), E192 (Adamson→Replogle RPE1 locked transfer; 175 tasks), E194 (family governance stress; 310 scenarios), E198 (12-protocol metric calibration on scPertEval arch1), E199 (263 K562 unseen-gene tasks), E200 (566 K562 full-context tasks), E158/E159 (PRESCRIBE official scores on Norman P3/P4) — all with frozen splits, prediction-first contracts and dual-remote sealing as described in the repository's experiment reports.

### Statistics

Spearman correlations and utilities carry 95% cluster-bootstrap CIs (5,000 draws; resampling perturbation conditions; for E201, jointly across targets). Paired comparisons are bootstrapped on the same clusters. Two-sided CIs; no multiplicity adjustment beyond pre-registered gate structure; negative results retained verbatim.

### Software and reproducibility

Analysis code, frozen contracts, seal manifests and hash chains reside in the SafeConf repository (GitHub: 1298020005/SafeConf; Gitee: librety/safe-conf). A pip-installable minimal audit package with a one-command reproduction of the E199/E200 main numbers is in preparation and will accompany the submission. Figures were generated from committed CSV tables by `agents/glm/paper/figure_scripts/make_figures.py`.

## References

1. Dixit, A. et al. Perturb-seq: dissecting molecular circuits with scalably single-cell RNA profiling of pooled genetic screens. *Cell* 167, 1853 (2016).
2. Roohani, Y. et al. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nat. Biotechnol.* 42, 972–980 (2024).
3. TxPert: using multiple knowledge graphs for prediction of transcriptomic perturbation effects. *Nat. Biotechnol.* (2026). doi:10.1038/s41587-026-03113-4.
4. Cui, H. et al. scGPT: toward building a foundation model for single-cell multi-omics and beyond. *Nat. Methods* 21, 1470–1480 (2024).
5. Cheng, J. et al. PRESCRIBE: predicting single-cell responses with Bayesian estimation. *Adv. Neural Inf. Process. Syst.* 38 (2025).
6. Ahlmann-Eltze, C. et al. Deep learning of perturbation responses does not outperform linear baselines. *Nat. Methods* (2025). doi:10.1038/s41592-025-02772-6.
7. Wei, A. et al. Benchmarking single-cell perturbation prediction (scPerturBench: 27 methods, 29 datasets). *Nat. Methods* (2025). doi:10.1038/s41592-025-02980-0.
8. PerturBench. *NeurIPS Datasets and Benchmarks Track* (2025).
9. Schäfer, P. et al. Towards principled evaluation of single-cell perturbation prediction models. bioRxiv 2026.07.23.740433 (2026).
10. Xing, E., Yau, H. & Wolf, G. Gaussian process modelling of single-cell perturbation data (GPerturb). *Nat. Commun.* (2025).
11. Alwani, A. & Wang, E. Y. ConfPert: distribution-free conformal coverage for single-cell perturbation predictors. ICML 2026 Workshop (OpenReview 1uE9rtYYzp, 2026).
12. Sui, P. et al. Medea: an omics AI agent for therapeutic discovery. bioRxiv 2026.01.16.696667 (2026).
13. Lahoti, P. et al. Responsible model deployment via model-agnostic risk estimation. *Patterns* 4 (2023).
14. Krogh, A. & Vedelsby, J. Neural network ensembles, cross validation, and active learning. *Adv. Neural Inf. Process. Syst.* 7 (1994).
15. 周老师组会追问 (2026-08); project evidence matrix `docs/实验结果/周老师问题_证据矩阵_20260801.md`. *(internal — remove from submission version)*
16. Molina, A. & Zhang, X. Perturbation response decomposition enables generalization to unseen contexts. bioRxiv 2026.07.24.740459 (2026). [also: PerturbMap, arXiv:2607.28090 (2026); HyperMap, bioRxiv 2026.04.23.720505 (2026).]
17. Lei, J. et al. Distribution-free predictive inference for regression. *J. Am. Stat. Assoc.* 113, 1134–1145 (2018).
18. Yang, Y. SafeConf: registered-family error certificates (project report, 2026-07-26). *(internal — merge or cite repository tag)*
19. Replogle, J. et al. Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq. *Cell* 185, 2559 (2022).
20. Mao, X. et al. Benchmarking virtual cell models for in-the-wild perturbation response. arXiv:2604.27646 (2026).

## Figure legends

**Figure 1 | SafeConf: post-prediction, fail-closed reliability contract.** **a**, Deployment scenario: a frozen family of predictors releases hashed predictions; SafeConf audits them using deployment-time signals only and outputs either a validated ranking (ROUTE) or an explicit abstention; target truth stays sealed until the audit artefacts are committed to two independent remotes. **b**, Three pre-registered gates adjudicated separately; failing a gate is a retained outcome, not grounds for re-tuning. **c**, Validation footprint: each signal carries the settings in which it is valid, untested, degraded or undefined; the setting decides whether a signal may be used at all.

**Figure 2 | Signal validity flips with the evaluation setting.** Spearman association with task error (95% cluster-bootstrap CIs) across settings: K562 unseen genes (E199), K562 full-context holdout (E200), preregistered cross-study RPE1 (E192, ABSTAIN by pre-registered gate), strict unseen genes (E158; PRESCRIBE official scores saturated → undefined), double-unseen (E189; negative utility), and the four-context blind audit (E201; sealed). The absence of a universally valid signal is the paper's central empirical finding and the motivation for the contract.

**Figure 3 | The two cleanest panels of the flip: 20% review-budget utility.** **a**, E199 (n=263): magnitude's utility is indistinguishable from random (CI crosses 0) while family disagreement delivers positive utility. **b**, E200 (n=566): predicted magnitude dominates every risk signal at the review budget. Both panels are reported as found; neither is averaged away.

**Figure 5 | Component footprint.** Per-signal Spearman association (bars, 95% CI) and 20%-budget review utility (blue diamonds) across all components in E199 (a) and E200 (b). No component is valid in both settings; the five-component equal-weight composite is diluted by its weak components.

**Figure 6 | Increment, mechanism, and the composite trap.** **a**, Paired increment of the risk score over predicted magnitude (E199 positive, E200 negative, with CIs). **b**, Rank-partial correlations: the informative axis switches from disagreement (E199) to magnitude (E200); magnitude's information survives controlling the risk score (0.875) while the converse is 0.301, and in E199 magnitude retains 0.005 after controlling disagreement. **c**, Equal-weight composites dilute magnitude where it dominates (E200) and help only mildly where it fails (E199) — exploratory, excluded from the E201 main analysis.

**Figure 7 | Conditional-contract simulation.** **a**, 20%-budget utility of three deployment strategies across nine frozen settings; the contract uses only gate-passing signals and abstains elsewhere (ABSTAIN = 0). **b**, Mean versus worst-setting utility: means are close, but only the contract guarantees a non-negative floor. Illustrative (in-sample footprint labels); E201 is the prospective test.

**Figure 4 | E201: four-context blind audit.** **a**, 4×4 retraining grid (leave-one-cell-line-out × seeds), target perturbed expression accessed zero rows throughout training. **b**, Frozen release pipeline; the irreversible truth-release step runs only after predictions, risk tables, baselines and hashes are committed to GitHub and Gitee. **c**, Four-target adjudication panels — [[E201-PLACEHOLDER-FIG4C: fill with per-target gate outcomes after STAGE_7; report all four targets]].

## Table 1 (draft, extends E202a): setting × method verdict ledger

| Setting (evidence) | Tasks | Signal / method | Association with error | 20% utility | Contract verdict |
|---|---:|---|---|---|---|
| K562 unseen genes (E199) | 263 | family disagreement | ρ=0.395 [0.283,0.497] | 0.208 [0.103,0.376] | ROUTING SUPPORTED |
| K562 unseen genes (E199) | 263 | predicted magnitude | ρ=0.096 [−0.026,0.219] | 0.040 [−0.083,0.226] | NOT VALID here (CI×0) |
| K562 context holdout (E200) | 566 | transfer risk | ρ=0.424 [0.351,0.495] | 0.365 [0.236,0.481] | correlates; magnitude-dominated |
| K562 context holdout (E200) | 566 | predicted magnitude | ρ=0.880 [0.844,0.909] | 0.913 [0.875,0.952] | DOMINANT baseline |
| K562 context holdout (E200) | 566 | source dispersion | ρ=0.664 [0.607,0.714] | 0.648 [0.544,0.734] | valid, subordinate |
| Cross-study RPE1 (E192) | 175 | family diversity | ρ=0.300 [−0.040,0.580] | 0.696 [0.113,0.872] | PREREGISTERED ABSTAIN |
| Strict unseen genes (E158/E159) | 48 | PRESCRIBE official combined / epistemic | undefined (saturated) | undefined | ABSTAIN (scores saturated) |
| Double unseen (E189/E191) | — | family disagreement | Spearman −0.349 to −0.241 (negative) | diversity −0.127; magnitude −0.080 (below random) | NEGATIVE — abstain |
| 4-context LOCO (E201) | 2,008 | SafeConf risk / magnitude | [[E201-PLACEHOLDER-T1]] | [[E201-PLACEHOLDER-T1]] | pending sealed evaluation |

## Data and code availability

Source datasets: public Perturb-seq studies as cited (Replogle/Adamson/Norman/Frangieh panels via scPertEval and original accessions; GSE225807). Analysis code, frozen contracts, seals and hash chains: GitHub `1298020005/SafeConf` and Gitee `librety/safe-conf`, branch `exp/task-risk-audit-20260611`. Minimal pip-installable audit package: in preparation; will be linked before submission.

## Acknowledgements, funding, competing interests, author contributions

[To be completed with the supervisor before submission. AI-assistance disclosure: drafting and figure-generation assistance by GLM/ZCode and earlier tools was supervised by the author, who verified all numbers against frozen tables.]
