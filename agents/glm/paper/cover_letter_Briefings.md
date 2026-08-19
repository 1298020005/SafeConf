# Cover Letter — Briefings in Bioinformatics（主投）

（GLM 起草 v1，2026-08-17；作者个人信息、日期与 E201 结果句待补）

---

Dear Editors,

We are pleased to submit our manuscript, **"SafeConf: a fail-closed
post-prediction reliability contract for single-cell perturbation models"**,
for consideration as an Original Article in *Briefings in Bioinformatics*.

Single-cell perturbation predictors (GEARS, CPA, scGPT, TxPert) are increasingly
used to plan CRISPR and chemical experiments, yet a deployed prediction arrives
with no statement of **which tasks a given uncertainty signal may be trusted
on**. Our manuscript addresses exactly this deployment gap with a
model-agnostic auditing layer that operates strictly after predictions are
issued and before target measurements exist.

We believe the work fits *Briefings in Bioinformatics* for three reasons:

1. **A concrete methods contribution with a blunt empirical message.** Across
   public TxPert retraining audits on K562, the validity of deployment-time
   risk signals flips with the evaluation setting: family disagreement ranks
   unseen-gene errors where predicted magnitude cannot (Spearman ρ = 0.395,
   95% CI 0.283–0.497, vs 0.096 with a CI crossing zero), while under
   full-context holdout a one-line magnitude baseline dominates everything
   (ρ = 0.880 vs 0.424; 20%-budget review utility 0.913 vs 0.365). We report
   both directions as found, and build the fail-closed contract (rank only
   inside a signal's pre-registered *validation footprint*; abstain elsewhere)
   as the operational answer.
2. **A fully pre-registered, verifiable blind protocol** for the central
   experiment: leave-one-cell-line-out across K562/RPE1/HepG2/Jurkat, four
   seeds, 2,008 tasks, with predictions, risk tables and baselines committed
   to two independent remotes before any target truth is opened.
   [[E201-RESULTS-SENTENCE: one line with the adjudicated outcome.]]
3. **Practical relevance.** The contract tells an experimental group which
   predictions may be triaged for review or wet-lab validation, and—critically
   —when the honest answer is "no existing signal is entitled to rank here",
   including cases where a competitor's official confidence scores saturate
   and become undefined on strictly unseen genes.

All data are public; analysis code, frozen contracts, seals and hash chains
are available (GitHub + Gitee), and a pip-installable one-command auditor
(`safeconf-audit`) independently recomputes the headline numbers from the
committed task-level tables. This manuscript is not under consideration
elsewhere; all authors have approved the submission.

Thank you for your consideration.

Sincerely,
[corresponding author, on behalf of all authors]

---

# Cover Letter — Bioinformatics（备投，Applications/Methods 轨道）

Dear Editors,

We submit **"SafeConf: a fail-closed post-prediction reliability contract for
single-cell perturbation models"** for consideration in *Bioinformatics*.

The manuscript contributes a model-agnostic, post-hoc auditing layer for
perturbation predictors, a pre-registered fail-closed decision contract with
three separated gates, and honest negative results: in the hardest deployment
setting we test (full cellular-context holdout), a one-line predicted-magnitude
baseline outperforms every risk signal at a fixed review budget — a finding we
report rather than tune away. [[E201-RESULTS-SENTENCE.]]

Per the journal's policy, source code is freely available at stable public
URLs (GitHub `1298020005/SafeConf`, Gitee `librety/safe-conf`), together with
a pip-installable minimal auditor (`pip install safeconf-audit`) whose single
command independently recomputes the E199/E200 headline statistics from
committed task tables.

Sincerely,
[corresponding author]
