# Cover letter draft

Dear Editors,

We submit **“Public perturbation evidence enables cold-start risk auditing for single-cell perturbation prediction”** for consideration as a research article in *IEEE/ACM Transactions on Computational Biology and Bioinformatics*.

The manuscript addresses a practical problem in single-cell perturbation modeling: a new predictor must be reviewed before its own screen-specific error labels are available. We introduce SafeConf, a post-hoc risk-auditing protocol that converts protocol-compatible public perturbation experiments into an auditable risk ranking. The protocol keeps public evidence, source-model errors, and target-model feedback in separate information budgets and evaluates each extension under the same task and review budget.

On a fixed McFaline benchmark, PublicRule recovered 22 and 24 of the 43 highest-error tasks for two predictors, compared with 4 and 5 for prediction magnitude. A retrospective GEARS/scGPT cross-family analysis provides stress evidence, while a frozen KOLF2.1J evaluation demonstrates the complete competence-gate, score-freeze, mixed-history, and fallback protocol. We report the broad external interval and retain competence failures and negative controls rather than presenting a universal superiority claim.

The contribution is a reproducible reliability layer and evaluation contract, rather than another upstream perturbation predictor. The manuscript includes source data summaries, task-level score hashes, information ledgers, figures, supplementary audit tables, and commands for reproducing all reported metrics without reopening the frozen confirmation truth.

This manuscript is not under consideration elsewhere. All authors have approved the submission and agree with its contents.

Sincerely,

AUTHOR_1
