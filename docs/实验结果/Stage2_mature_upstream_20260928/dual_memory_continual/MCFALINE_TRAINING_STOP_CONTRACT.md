# McFaline upstream registered validation-only compute-budget amendment

Registered after observing training throughput and the partial validation-loss
trajectory, but before the stopping epoch and before any McFaline test outcome
was opened. It is therefore an engineering compute-budget amendment, not a
claim that the epoch budget was preregistered before training.

## Problem

Both official candidate configurations use `max_epochs=500` and validation
early-stopping patience `50`. With the official 8000-cell batches and a
single-process HDF5 reader (the only configuration that passed the memory
preflight), one epoch takes about 16--18 minutes. The official patience would
consume most of the one-week study budget and the 44-hour process guard could
terminate training before validation evaluation.

Using 12 loader workers was already rejected in a recorded preflight because
worker processes duplicated HDF5 state and were OOM-killed before a GPU batch
was produced. Restarting training for throughput tuning would discard valid
checkpoints and spend additional GPU hours.

## Registered engineering decision

- Continue both existing runs without restart until **epoch 15 has completed**.
- Select the single checkpoint retained by the official validation-loss
  `ModelCheckpoint` callback. Test outcomes do not participate.
- At epoch 15 completion, stop the training processes, record validation-loss
  histories and checkpoint hashes, and let the existing supervisor perform
  validation-only competence evaluation.
- Both candidates use exactly the same epoch budget. Each must first pass the
  registered validation competence gate. If both pass, select the candidate
  with lower official validation macro RMSE. If their relative macro-RMSE
  difference is below 1%, select the lower measured training cost; if the cost
  measurement is tied or unavailable, select DecoderOnly because its registered
  architecture has fewer trainable parameters. This rule is fixed before any
  SafeConf result is computed.
- If neither candidate passes competence, do not open McFaline test truth.
- The McFaline test partition remains sealed.

This changes the compute budget only. It does not change architecture,
optimizer, batch size, upstream selection rule, SafeConf method, or evaluation
metric.
