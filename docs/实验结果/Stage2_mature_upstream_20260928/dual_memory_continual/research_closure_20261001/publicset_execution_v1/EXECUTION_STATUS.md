# SafeConf execution checkpoint

This checkpoint contains actual experiment outputs, executable implementation, and a TCBB manuscript draft. The experimental goal remains active while physical-content controls and the aligned-public feedback rerun complete. The user has deferred manuscript/PDF work until the model is selected; existing drafts are archived, and no further writing or compilation is scheduled.

## Completed

- Public biology: 120 unique NN fitting stages across Source/McFaline, five gene folds, two neural builders, three seeds. Interrupted attempts and stage/model reuse are recorded; the prototype is included in this count.
- Public comparison: support mean, original HGB builder, pointwise network, DeepSets; aligned cell-weighted McFaline history and unchanged query truth; 5000 paired gene bootstrap and stronger prediction-time matching rules.
- Frangieh: scGPT-to-GEARS and reverse, fixed five gene risk folds inside each original checkpoint fold; 60 Ridge plus 60 HGB fits, native512, retrospective SEEN. One saved-model monotonic-output repair completed without refitting.
- Independent scope/implementation checks, complete manuscript draft, figures and an actually compiled manuscript PDF.

## Scientific decision so far

The DeepSets context branch does not establish a gain over pointwise weighting. Pointwise weighting improves Source references relative to the original builder, while the strongest simple prediction/metadata rules remain essential comparators. The registered joint comparison has now completed: 400 context-scoped HGB fits and 5000 paired gene resamples. Neither neural builder passes replacement against B0/B1 in both directions; the DeepSets extra branch passes neither direction against pointwise. These are paired, fixed-comparison decisions, not rankings of the highest score.

## Completed closure

- Registered nested reader: 400 risk fits, 240 nested neural stages, 5000 paired gene bootstrap.
- Physical content control: 1200 risk fits, 500 frozen prior sets, 5000 paired gene bootstrap; model/donor/prior integrity passed.
- Reference headroom diagnostic: 2350 DEV queries, zero new model fits.
- Aligned-public feedback: 45 affected fits, fixed212-task/152-gene evaluation, 5000 paired bootstrap; preserved17 inputs and unaffected scores checked.
- Experimental model selected in FINAL_MODEL_SETTINGS.json; decisions in FINAL_EXPERIMENTAL_DECISION.md.
- Continual release finite-value repair and specific tests passed; manuscript/PDF work remains deferred.

## Reliable process tracking

Long experiments are submitted through user systemd services. Actual unit state and MainPID, not stale JSON state alone, establish whether a job is running. Cached complete models/stages survive a session and are reused on recovery.

Read `risk_followup_v1/registered_universal_v1/` for the corrected reader results; old relative-sparsity reader fits are retained and do not stand in for the registered Universal-P version.
