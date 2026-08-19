26 July 2026

Editors<br>
*BMC Bioinformatics*

Dear Editors,

Please consider our Research Article, **“SafeConf: registered-family error certificates for auditable single-cell perturbation prediction.”**

Single-cell perturbation models now generate large collections of counterfactual expression predictions, but users still lack a clear post-hoc statement about how wrong a frozen heterogeneous family of models may be on an unseen perturbation. SafeConf addresses this deployment problem by combining three auditable components: deterministic lower error certificates from registered-family geometry, target-cluster split-conformal upper events transferred through an explicit centroid-shift penalty, and a fail-closed release sequence that separates prediction, calibration, and evaluation truth.

We evaluated a fixed family of five scGPT and five GEARS fits in four human CRISPR perturbation studies containing 2,433 held-out tasks and 737 target clusters. The deterministic certificates had zero numerical violations. Family upper bounds covered 666/737 targets in a retrospective pooled description. Importantly, a fully preregistered independent K562 RBP study covered 16/20 targets and failed its registered 17/20 success rule. We retained that negative outcome, all thresholds, and all target identities. A separate 8,777-instance stress audit covered random missing pairs, unseen contexts, unseen perturbations, double-unseen tasks, and two direct dataset transfers. It showed that deterministic validity persists while tightness changes with the failure structure; no upper claim was transported without target-domain calibration. A standard-library validator reconstructs the main registered results in 12,033 checks with zero failures.

The manuscript is relevant to *BMC Bioinformatics* because it provides a reusable statistical and software framework for uncertainty auditing in computational genomics, accompanied by public task-level results, target-level summaries, source locks, and validation code. Model-specific diagnostics, nested adaptive-baseline comparisons, and a native-endpoint PRESCRIBE analysis define what the certificate does and does not replace. The work distinguishes exact deterministic statements from empirical coverage and discusses the limits imposed by shared model bias, finite calibration samples, and dataset shift.

This manuscript is original, has not been published, and is not under consideration elsewhere. All authors will approve the submitted version. The authors declare no competing interests. **[Confirm these statements and insert the corresponding author's contact details before submission.]**

OpenAI Codex assisted with code auditing, numerical cross-checking, document organization, and language drafting. The authors verified the source evidence, citations, calculations, and manuscript text and retain full responsibility for the submitted work. This assistance is disclosed in the Acknowledgements.

Thank you for your consideration.

Sincerely,

Yifan Yang<br>
School of Computer and Information Engineering<br>
Henan University<br>
Kaifeng, Henan, China<br>
**[Corresponding email]**
