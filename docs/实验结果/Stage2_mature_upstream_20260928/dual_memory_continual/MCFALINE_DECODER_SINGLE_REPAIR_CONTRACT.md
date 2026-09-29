# McFaline DecoderOnly single upstream repair contract

Registered after raw DecoderOnly validation competence and before computing any
repaired score. McFaline test expression remains sealed.

## Observed failure

On the fixed 512-gene effect contract, DecoderOnly has macro RMSE `0.022477`
versus `0.022033` for the strongest registered simple baseline
(`train_state_mean_effect`): relative gap `+2.0145%`. Two of three validation
strata are within the 2% non-inferiority margin, and the perturbation-cluster
bootstrap interval `[+1.3771%, +2.6443%]` does not support a stable disadvantage
beyond 2%. The raw candidate nevertheless fails the strict point-estimate rule.

Epoch-14 and epoch-15 official validation losses are both `0.040603`, so adding
training epochs has no registered empirical justification.

## Single repair hypothesis

The published DecoderOnly prediction may contain task-specific signal while
remaining slightly over-dispersed relative to the strong state-mean prior. A
validation-only convex shrinkage can retain that signal and reduce variance:

```text
effect_repaired = alpha * effect_decoder
                + (1-alpha) * effect_train_state_mean
```

This is upstream calibration. It cannot use any SafeConf score, risk feature,
or McFaline test value.

## Frozen repair

- Five perturbation-cluster folds fixed by SHA-256 of perturbation identity.
- Candidate alphas: `{0.25, 0.50, 0.75, 1.00}`.
- Alpha is selected independently inside each outer-train partition by lowest
  task-level effect RMSE; ties favor the larger Decoder weight.
- The OOF repaired effect is evaluated once on each outer-test partition.
- Final test-time alpha is selected with the same rule on all validation tasks
  only after the OOF competence decision.
- 5,000 paired perturbation-cluster bootstrap draws.

The lower bound `alpha >= 0.25` prevents the repair from relabelling the simple
baseline as a competent published upstream.

## Pass rule

The repaired upstream passes only if all conditions hold:

1. OOF repaired macro RMSE is no higher than the strongest simple baseline;
2. at least 60% of validation strata are within the 2% margin;
3. the paired relative-gap bootstrap lower bound does not exceed +2%; and
4. every selected fold alpha is at least 0.25.

If this repair fails, McFaline test remains sealed and no second upstream repair
or third upstream is attempted this week.
