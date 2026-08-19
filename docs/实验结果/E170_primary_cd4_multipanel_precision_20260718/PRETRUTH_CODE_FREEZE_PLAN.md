# E170 pretruth code freeze and execution order

## Immutable order

1. Commit and push the E170 asset builder, pretruth runner, and joint postgate evaluator to GitHub and Gitee.
2. Build all P01–P04 `F2_pretruth` bundles. Test-donor targeting X and all column-unseen train/validation X remain unread.
3. Train scGPT and GEARS with seeds 3407/3408/3409 for every panel. Complete all four panels even if an earlier panel gate fails; panel dropping and optional stopping are forbidden.
4. Commit every pretruth release and PASS/FAIL snapshot together, then push that gate commit to both remotes.
5. Only if all four registered gates PASS, build the four `F3_postgate` bundles from the 1,200 registered test rows per panel.
6. Run the already-frozen joint evaluator on E170's 800 new targets only. E168's 200 targets are excluded from the new confidence interval and permutation test.

## Frozen commands

```bash
python tools/scripts/build_e170_primary_cd4_panel_assets.py --panel P01 --stage pretruth
python tools/scripts/run_e170_primary_cd4_panel_pretruth.py --panel P01 --device cuda:0

python tools/scripts/build_e170_primary_cd4_panel_assets.py --panel P01 --stage postgate \
  --gate-snapshot docs/实验结果/E170_primary_cd4_multipanel_precision_20260718/pretruth_release/P01/PRETRUTH_GATE_SNAPSHOT.json \
  --gate-commit <all-panel-gate-commit> --branch exp/task-risk-audit-20260611

python tools/scripts/run_e170_primary_cd4_joint_postgate.py \
  --gate-commit <all-panel-gate-commit> --branch exp/task-risk-audit-20260611
```

P02–P04 use the same commands with the panel identifier changed. Each runtime attestation records the exact code head, builder hash, source SHA-256, row counts, and remote heads.

## Interpretation lock

E170 is an outcome-informed follow-up designed after the negative E168 result, but its 800 targeting-expression outcomes were not opened during design. It is a fresh-target precision replication within the same donor and study. It cannot be described as an independent donor cohort, a new wet-lab experiment, or a guaranteed journal-level result.
