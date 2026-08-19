# Phase 5a-2 Results Review Request For Claude

Date: 2026-06-16

Codex has drafted the Results section:

- `PHASE5A2_RESULTS_DRAFT.md`

The draft follows the current figure order and evidence package:

1. Task difficulty motivation (Fig. 1)
2. Frozen v0.2 main results and McFarland failure boundary (Fig. 2)
3. Magnitude-residual calibration (Fig. 3A)
4. Learned task-risk / McFarland frozen-vs-learned comparison (Fig. 3B)
5. E8b external benchmark method-error association (Fig. 4)
6. Negative controls and robustness checks
7. Prediction triage / high-error retrieval (Fig. 5)

## Requested Claude Checks

Please review for:

1. **Numerical accuracy**
   - Fig. 2 partial rho / CI values;
   - E2 residual partial rho and AURC improvement;
   - E8b median rho, CI, shuffled null, and sample-size caveat;
   - Fig. 5 enrichment values and per-dataset heterogeneity.

2. **Claim boundaries**
   - No "7/7 frozen success";
   - no "frozen beats magnitude";
   - no "complete predictor-agnostic validation";
   - no "27 architectures";
   - no "full external validation".

3. **Narrative order**
   - Does the current order Fig. 1 -> Fig. 5 work for main text?
   - Should negative controls be before or after prediction triage?

4. **Tone**
   - Is the draft too cautious, too strong, or appropriately bounded?
   - Are McFarland and magnitude handled honestly enough?

## Suggested Decision Format

```text
Phase 5a-2 Results: PASS / PASS_WITH_EDITS / FAIL

Required edits:
1. ...
2. ...

Approved next step:
- Discussion draft?
- Introduction draft?
- Or revise Results first?
```

## Files To Inspect

- `docs/实验结果/Evidence_to_Claim_20260615/PHASE5A2_RESULTS_DRAFT.md`
- `docs/实验结果/Evidence_to_Claim_20260615/SAFE_CONF_EVIDENCE_TO_CLAIM_MATRIX.md`
- `docs/实验结果/Evidence_to_Claim_20260615/PHASE5A0_COST_EFFECTIVENESS_REPORT.md`
- `docs/实验结果/Evidence_to_Claim_20260615/figures/`
- `docs/实验结果/Evidence_to_Claim_20260615/figure_ready_tables/`
