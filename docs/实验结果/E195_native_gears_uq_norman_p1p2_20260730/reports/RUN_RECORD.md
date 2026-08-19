# E195 run record

- completed: 2026-07-30T14:04:51+08:00
- Python: `/home/yyf/.conda/envs/scgpt_env/bin/python`
- final status: `COMPLETE`
- bootstrap: 5000
- git head: `799b51cce29146d8695a44ffc8759aaaf2338ce9`
- E195 code paths clean: `True`
- unrelated dirty entries recorded by hash: `fcbba1e2626e9f7113e2bfb08881553e838f61983495f9df812c5b216e7d9b9d`
- local raw-artifact entries: 60

## Training commands

### P1 seed 11

- elapsed seconds: 375.54
- child status: `ok`
- initial model hash: `b557928260ded34f6886e47b690c48636f4ed161cdb27e17e6a321e4ffa83249`
- trained model hash: `611311594478492cd344996a56556e8138bbb099b9fe1c447b747606aab029a5`

```bash
/home/yyf/.conda/envs/scgpt_env/bin/python -m safetrans_confidence.cli.run_gears_prediction_records --dataset norman --seed 11 --split single --run-type formal --epochs 10 --hidden-size 48 --decoder-hidden-size 16 --num-similar-genes 10 --batch-size 32 --test-batch-size 64 --max-cells-per-condition 32 --condition-sampling-seed 20260766 --device cuda:0 --out-dir /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P1/raw_gears/seed_11 --data-path /home/yyf/data/safeconf_e195_gears_strict_v1/P1/seed_11 --test-perturbations-file /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P1/E195_FROZEN_MANIFEST.csv --fixed-test-deterministic-val --train-gene-set-size 0.75 --max-genes 6000 --lr 0.001 --weight-decay 0.0005 --coexpress-threshold 0.4 --direction-lambda 0.1 --uncertainty --strict-score-lock-before-truth --require-cuda
```

### P1 seed 22

- elapsed seconds: 375.56
- child status: `ok`
- initial model hash: `aab65d36a6443b7fa3f8886ea1456c2b11d09310849b9ca565658df5d71538bd`
- trained model hash: `449999aedc772d566a5309e71172a84eb1801b30410f557c3011a97f65023d37`

```bash
/home/yyf/.conda/envs/scgpt_env/bin/python -m safetrans_confidence.cli.run_gears_prediction_records --dataset norman --seed 22 --split single --run-type formal --epochs 10 --hidden-size 48 --decoder-hidden-size 16 --num-similar-genes 10 --batch-size 32 --test-batch-size 64 --max-cells-per-condition 32 --condition-sampling-seed 20260766 --device cuda:1 --out-dir /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P1/raw_gears/seed_22 --data-path /home/yyf/data/safeconf_e195_gears_strict_v1/P1/seed_22 --test-perturbations-file /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P1/E195_FROZEN_MANIFEST.csv --fixed-test-deterministic-val --train-gene-set-size 0.75 --max-genes 6000 --lr 0.001 --weight-decay 0.0005 --coexpress-threshold 0.4 --direction-lambda 0.1 --uncertainty --strict-score-lock-before-truth --require-cuda
```

### P1 seed 33

- elapsed seconds: 371.40
- child status: `ok`
- initial model hash: `5699561dec2ffd521c29f9bee9240a415830a0c2283e92b3954f93e2e0bb8745`
- trained model hash: `2e9756dd0c84bc19a343e133d99b28615eb8d163128034b12e2762b4f59bcdbb`

```bash
/home/yyf/.conda/envs/scgpt_env/bin/python -m safetrans_confidence.cli.run_gears_prediction_records --dataset norman --seed 33 --split single --run-type formal --epochs 10 --hidden-size 48 --decoder-hidden-size 16 --num-similar-genes 10 --batch-size 32 --test-batch-size 64 --max-cells-per-condition 32 --condition-sampling-seed 20260766 --device cuda:0 --out-dir /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P1/raw_gears/seed_33 --data-path /home/yyf/data/safeconf_e195_gears_strict_v1/P1/seed_33 --test-perturbations-file /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P1/E195_FROZEN_MANIFEST.csv --fixed-test-deterministic-val --train-gene-set-size 0.75 --max-genes 6000 --lr 0.001 --weight-decay 0.0005 --coexpress-threshold 0.4 --direction-lambda 0.1 --uncertainty --strict-score-lock-before-truth --require-cuda
```

### P2 seed 11

- elapsed seconds: 371.42
- child status: `ok`
- initial model hash: `b557928260ded34f6886e47b690c48636f4ed161cdb27e17e6a321e4ffa83249`
- trained model hash: `c19338eed19c766e0fc69a217fc336fc41c1df387a69d5ccacdac144d221b407`

```bash
/home/yyf/.conda/envs/scgpt_env/bin/python -m safetrans_confidence.cli.run_gears_prediction_records --dataset norman --seed 11 --split single --run-type formal --epochs 10 --hidden-size 48 --decoder-hidden-size 16 --num-similar-genes 10 --batch-size 32 --test-batch-size 64 --max-cells-per-condition 32 --condition-sampling-seed 202607752 --device cuda:1 --out-dir /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P2/raw_gears/seed_11 --data-path /home/yyf/data/safeconf_e195_gears_strict_v1/P2/seed_11 --test-perturbations-file /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P2/E195_FROZEN_MANIFEST.csv --fixed-test-deterministic-val --train-gene-set-size 0.75 --max-genes 6000 --lr 0.001 --weight-decay 0.0005 --coexpress-threshold 0.4 --direction-lambda 0.1 --uncertainty --strict-score-lock-before-truth --require-cuda
```

### P2 seed 22

- elapsed seconds: 370.43
- child status: `ok`
- initial model hash: `aab65d36a6443b7fa3f8886ea1456c2b11d09310849b9ca565658df5d71538bd`
- trained model hash: `1e85fb7389d079883d6c732bca9d9ab8738df363179554359ad766f61aab8708`

```bash
/home/yyf/.conda/envs/scgpt_env/bin/python -m safetrans_confidence.cli.run_gears_prediction_records --dataset norman --seed 22 --split single --run-type formal --epochs 10 --hidden-size 48 --decoder-hidden-size 16 --num-similar-genes 10 --batch-size 32 --test-batch-size 64 --max-cells-per-condition 32 --condition-sampling-seed 202607752 --device cuda:0 --out-dir /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P2/raw_gears/seed_22 --data-path /home/yyf/data/safeconf_e195_gears_strict_v1/P2/seed_22 --test-perturbations-file /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P2/E195_FROZEN_MANIFEST.csv --fixed-test-deterministic-val --train-gene-set-size 0.75 --max-genes 6000 --lr 0.001 --weight-decay 0.0005 --coexpress-threshold 0.4 --direction-lambda 0.1 --uncertainty --strict-score-lock-before-truth --require-cuda
```

### P2 seed 33

- elapsed seconds: 370.45
- child status: `ok`
- initial model hash: `5699561dec2ffd521c29f9bee9240a415830a0c2283e92b3954f93e2e0bb8745`
- trained model hash: `ce0416d71ae8a3d64ec23902a55a38b44edb7e995da034c3724c26ddc570c7a4`

```bash
/home/yyf/.conda/envs/scgpt_env/bin/python -m safetrans_confidence.cli.run_gears_prediction_records --dataset norman --seed 33 --split single --run-type formal --epochs 10 --hidden-size 48 --decoder-hidden-size 16 --num-similar-genes 10 --batch-size 32 --test-batch-size 64 --max-cells-per-condition 32 --condition-sampling-seed 202607752 --device cuda:1 --out-dir /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P2/raw_gears/seed_33 --data-path /home/yyf/data/safeconf_e195_gears_strict_v1/P2/seed_33 --test-perturbations-file /home/yyf/proj/docs/实验结果/E195_native_gears_uq_norman_p1p2_20260730/panels/P2/E195_FROZEN_MANIFEST.csv --fixed-test-deterministic-val --train-gene-set-size 0.75 --max-genes 6000 --lr 0.001 --weight-decay 0.0005 --coexpress-threshold 0.4 --direction-lambda 0.1 --uncertainty --strict-score-lock-before-truth --require-cuda
```

## Output hashes

The run record excludes itself from the hash manifest to avoid a self-referential hash cycle.

| path | bytes | sha256 |
| --- | --- | --- |
| .gitignore | 61 | 5fe2e4a6498658ffafb1c81f49fb2bf5c39c3d1fca8e3be9a10a1ea33277caef |
| ANALYSIS_FREEZE.md | 13649 | 384d072fefbe49f3f962c27281e7257cc9d462c59a3edee8611ca0aa4d23c55f |
| E195_STATUS.json | 1613 | 616e13e6bd11a895b3df5dc8e6f85576dca40c579acbbe720021855b20bba999 |
| figures/E195_VISUALIZATION_PROVENANCE.json | 967 | e855b9848cf15aba6c8292d72255400b81ac4ec70c7ee8cf1b429dbc6a8e727d |
| figures/E195_native_uq_comparison.pdf | 35946 | 580930838582b9f83fe51429a47bbb4ab6e5487060c570d835f2ac93036f611c |
| figures/E195_native_uq_comparison.png | 449711 | b770a614ab019a7a969c2ae155eff7267ccd076b426035c245c2968fd9f1c6b1 |
| figures/E195_same_prediction_summary.pdf | 30995 | 86a0102d1d4cf8cab5a572b6c84082e023eafb1745fc3f20e5f3937fa54fa643 |
| figures/E195_same_prediction_summary.png | 612071 | 36698869555d0f0214967c3ca4f178dddedef8d2bc12331b916f0c9e8ae8dbfc |
| panels/P1/E195_FROZEN_MANIFEST.csv | 3280 | f1162e8378fa186153b393b9e3e2a7d5a99189f44e7e0afc6f079d76677e565a |
| panels/P2/E195_FROZEN_MANIFEST.csv | 4226 | 36597e0cf025948598bc2195e34e4dd87517be38e7dc3c35bcc9fc05c42df8db |
| reports/E195_INTERPRETATION.md | 819 | eb1442a439fc9d5620e4e3c0b034bc3a113ef2b6ef7f312ad3cd5a54f79b10f6 |
| reports/E195_REPORT.md | 6409 | b9d02a68da04b0f9fd3869c5d0bdbb79dd0f1fefd945f3481ffbca5288184766 |
| reports/E195_RESULT_README.md | 3561 | 74c9b91c79e2019fe0be13b59b22358331fbc2754977f6110a30eed149c2d0fc |
| tables/E195_ASSOCIATION.csv | 7517 | 2b751f974e742a8b9b0477b4b121e36287ff632138589b128497866e6569678e |
| tables/E195_DYNAMIC_RANGE_AUDIT.csv | 1960 | 233176cc507d39517d9f40e978a87b6293ed1b7786cb71e803abb469e5998960 |
| tables/E195_FAMILY_TASKS.csv | 10985 | b244425038c6edb9ad8d06913a3783bf599d11e28c4c193dac9f919bcc31d95e |
| tables/E195_INPUT_HASHES.csv | 3066 | bda2658d957e85f757ff96cca58a258d0747e46de5785ad93480cbf141d82d12 |
| tables/E195_INVARIANT_AUDIT.csv | 6058 | 0fb957c7dc07d6e0889059671ecf888bb7b2bc431fe20e67e20018c6bee77069 |
| tables/E195_PAIRED_SCORE_DELTAS.csv | 3371 | bac13c8a0c0a2db8d9f2580fcf1c6490816ece31b81b46b94cb567c6cfb85108 |
| tables/E195_RAW_ARTIFACT_HASHES.csv | 18150 | 7c45a8e63477a4f3974d013dec28a84ab1769ad75a4b6f3f5cc2944ed90a43ab |
| tables/E195_RISK_COVERAGE.csv | 44169 | 9b47eba9b3094d8416ce48ec109f2487a96d20efc1a2d93201fc32ca36f6bfd7 |
| tables/E195_ROUTING_METRICS.csv | 21028 | 4d4c67849a5acb1c600a8d9b9163ef90177a22ec4f2160f7970d53d5712005fd |
| tables/E195_RUNTIME_ENVIRONMENT.csv | 245 | a885087718109b885b8b29dcf61a2667155bfaac044bde1e16bff94d3b3d1ff2 |
| tables/E195_SCORE_LOCK_AUDIT.csv | 717 | 10dff474cafe73107a07c7d46f3bd1c5db9223e4d883a076d1e730158139a241 |
| tables/E195_SINGLE_SEED_TASKS.csv | 45785 | aef8ec88d484af088fcde5712ac0d49065aa27bd9449b0e48ddbb1c1713466af |
| tables/E195_SUPPORT_EXPOSURE_AUDIT.csv | 13376 | ed88f0401f533f92213f2fc62928935d37a8fb730f040bd78f0fcb8a7022da2c |
| tables/E195_SYSTEM_COMPARISON.csv | 2351 | 9729a6796bf8875dfa4aeff3d66c7dc713091a2a33cf8ab36f21bc99b18929b1 |
