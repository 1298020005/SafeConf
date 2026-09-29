# McFaline upstream preflight repair

## Observed failure

Both official candidates reached model construction, CUDA initialization and the first training batch request. Before either GPU executed a batch, PyTorch reported `DataLoader worker ... killed by signal: Killed`.

- LatentAdditive: official batch size 8000, 12 workers.
- DecoderOnly: official batch size 8000, 12 workers.
- GPUs remained at 3 MiB after failure.
- Host memory returned to more than 100 GiB free when worker processes exited.

This is a data-loading process-memory failure, not an upstream competence result and not a GPU out-of-memory result.

## Single registered repair

Set `data.loader.num_workers=0` for both candidates. Preserve:

- official batch size 8000;
- model architecture and width;
- optimizer and learning rate;
- validation split and all biological data;
- test truth sealed.

The repair prevents 12 workers per candidate from duplicating HDF5/AnnData state. No SafeConf result was available or used.
