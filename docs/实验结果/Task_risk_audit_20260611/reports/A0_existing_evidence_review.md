# A0 existing evidence review

Date: 2026-06-11

## What is already answered

- Corrected seven-main frozen v0.2 remains the interpretable baseline. It is useful, but it is not the strongest current method layer.
- McFarland is still a frozen-v0.2 failure boundary: frozen partial rho = -0.061.
- The learned LODO reliability layer rescues McFarland on the corrected run: LODO partial rho = 0.162.
- LODO is positive in 7/7 corrected main datasets. This is the cleanest dataset-transfer evidence.
- LOPO unseen-predictor probes are positive in 14/14 dataset-by-predictor checks. This supports predictor transfer, while still sharing task-level features with the training predictors.
- External validation has positive partial rho in 4/4 studies, but several external AURC intervals are uncertain; write it as supportive evidence, not as four conclusive wins.

## What A0 does not answer

A0 does not directly say whether V0StrongBaseline and ContextSimBaseline fail on the same tasks. That requires A1: a paired same-task error audit.

## Working interpretation before A1

SafeConf should currently be described as an external task-risk / prediction-risk protocol with a learned reliability layer. Avoid claiming that frozen v0.2 alone is a complete model-specific reliability evaluator.
