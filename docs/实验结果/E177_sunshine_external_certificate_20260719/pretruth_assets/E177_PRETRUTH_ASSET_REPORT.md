# E177 pretruth asset build

F2 asset build is complete. The asset directory is outside Git:

`/home/yyf/data/safeconf_e177_external/isolated/F2_pretruth`

What was read:

- exact controls: 2,500 rows
- train target rows: 8,146 rows
- validation target rows: 1,193 rows

What stayed sealed:

- calibration target rows read: 0
- evaluation target rows read: 0

The source H5AD stores `X` as a backed CSC matrix. The audit therefore records logical exported rows, not a physical HDF5 row-level claim. This is acceptable for the current computational validation because no calibration or evaluation target vectors are exported into F2.

F2 outputs:

- 512-gene model panel, including all 144 registered targets
- 8 same-group control profiles
- 512 train/validation task effect vectors
- 640 query-only tasks without `y`
- control-only coexpression graph for GEARS

Next gate: train the five-seed scGPT/GEARS families and run the truth-blind stability checks before opening calibration truth.
