# Source objective probe: one technical repair

Status: **V2 PREPARED; NOT RUN**. Root's approved v1 stopped after one rank fit. The failed runtime output is preserved. The original script, README, and ownership manifest are preserved byte for byte in `failed_preparation_v1`; the original README remains a historical preparation snapshot.

Root's independent review reports that the first fold's clipped risk replay passed (`max_abs_difference = 1.11e-16`), while the bundled float64 truth comparison failed because float32 cached errors had been serialized to decimal CSV and parsed as float64 (maximum gap about `3.7e-9`). Restoring those archive decimal values to float32 recovered the original bits. This repair addresses that codec check; it does not change the method or relax the risk tolerance of `1e-14`.

The repaired script requires cached Source error dtype `float32`, converts the matched Source archive decimal errors back to float32, and checks exact `uint32` bit equality. All arm/reference metrics then use the matching original archive decimal errors, preserving the frozen metric population and codec. Training labels, affine transforms, training weights, and features continue to use the original cached Source training rows. No evaluation errors enter label transforms or fitting.

Exactly one existing model may be reused: `approved_20261002_v1/MODEL_GAT_to_Exphormer_fold0_rank.joblib`. Before any model load, the script checks the fixed pins below, verifies v1 was failed after one fit with all inputs unchanged, and requires all current Source, archive-reference, and core-helper SHAs to equal v1's registered input SHAs. There is no model search, automatic discovery, or other reuse. The reused model is copied byte for byte into the fresh v2 output.

| Fixed input | SHA256 |
| --- | --- |
| v1 first-fold rank model | `37e378db03fe7949a1ffc8ed400b480fb6c6f580be26bccc10f08e3b8f271097` |
| v1 STATUS.json | `5fea8cc4c8fd32cb3034068f1ec7b18bf8a24ba6b3406d0bcc92ed6969e13119` |
| v1 REGISTRATION.json | `7959e8ff1d0123dd4c51cbeac0e39b8b0d6eb90082863dac2f48b89e4d03ca50` |
| preserved original script | `f77968fb5537c850b9812e58cd8d69eda904255babcec1a210eeddd3185be496` |

The cumulative budget stays **20 actual fits**: one already executed in v1 plus at most **19 new v2 fits** (nine rank and ten affine). V2 evaluates 20 models and saves 20 model files, including the reused file. `FIT_COSTS.csv`, `SCORE_FREEZE.json`, and `STATUS.json` distinguish model evaluations, new fits, previous fits, reused models, and cumulative fits. The three fixed references still add zero fits. The single-thread recipe and 1,200-second run-phase alarm remain fixed; at most 3,800 new boosting iterations are permitted. Outputs require a fresh runtime child and a fresh approval bound to the repaired script and exact repaired `SCOPE`; the approval also requires top-level `max_new_fits = 19` while `max_fits = 20` remains cumulative. Root handles authorization and the actual run.

Repaired script SHA256: **`6b6863f375bdf997f55fe320ae80ef11eeb6af907dd326b4089c042d14a95dc9`**.

The synthetic `--self-check` passed again: exact float32 codec bit restoration; float64 cache rejection; unchanged mathematical rank/raw reversal; missing approval rejection before Source/archive/hash/fit access. It performed **0 real Source reads, 0 actual fits, and 0 writes**. Syntax and whitespace checks passed. Preparation inspected only v1 registration/status metadata and hashed the fixed saved model bytes; it did not deserialize that model or execute v2.

This remains a point/fold DEV diagnostic. No bootstrap, CI, superiority, root-cause, formal score, or external evaluation is added.
