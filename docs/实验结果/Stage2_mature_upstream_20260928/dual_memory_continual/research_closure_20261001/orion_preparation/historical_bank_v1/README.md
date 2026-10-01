# Published Source history expansion

The completed implementation uses `build_safeconf_source_historical_bank_sorted_csr_agent.py` and the immutable contracts in `registered_v3_sorted_csr/`. The preliminary contracts and the failed original run remain available for provenance. The original run stopped on unsorted CSR indices before committing a bank. The separate revision sorts paired indices and values within bounded blocks and continues to reject duplicate gene entries.

The separate bank keeps all 2008 original metadata, effect and control rows exactly and appends only single-gene historical units whose target is absent from the original 580 eligible genes. The original 575 risk-training genes are a subset of those 580, so their historical retrieval groups cannot change. The seven original Source models are retained without fitting.

The processed Source cache is published Replogle 2022 and Nadig 2025 development biology. Its filename `de_adata_test.h5ad` does not denote Orion TEST. Its documented expression coordinate is already cellwise log1p(CP4000); the builder does not normalize or log it again. Each new unit uses the arithmetic mean across its observed cells and observed controls from its own study, context and batch, weighted by the unit's cell counts in each batch.

Only K562, RPE1, HepG2 and Jurkat are allowed. K562 Adamson and nonsingle-gene conditions are excluded. New units require at least 30 historical cells. The full measured 3352-gene axis contains the fixed 3285 endpoint genes exactly. Sparse zeros belong to this measured axis; missing genes fail. Guide, plate and replicate quality fields remain missing. A Source batch count does not establish independent biological replicates.

The metadata plan appends 3357 units for 1490 new genes. Already seen Orion metadata supports 144 distinct TEST gene clusters (107 HCT116 tasks and 125 HEK293T tasks; 88 genes occur in both contexts). This is a prospective bank revision chosen for metadata coverage, not the original confirmation. Orion expression and TEST truth are not inputs to the builder. Actual all-method finite coverage is checked separately before any TEST opening.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/scripts/build_safeconf_source_historical_bank_sorted_csr_agent.py build
```

The builder verifies its immutable contracts and the complete Source file SHA before reading X. It reads exact allowed contiguous CSR row runs with limits of 2048 rows and eight million nonzeros per chunk, a 4 GiB resident memory cap and a 1200 second aggregation limit. Outputs are staged and committed only after exact legacy prefix checks, the Source compatibility diagnostic, coverage and original input hash rechecks. The completed bank is under `/home/yyf/runtime_artifacts/safeconf_research_20261001/public_source_history_expanded_20261002_v2`.

The actual build completed 571090 authorized Source cells in 551 chunks. Its largest legacy effect gap was 4.1649e-6 and control gap was 9.0586e-7, both below the fixed 1e-5 tolerance. The final legacy metadata/effect/control prefixes are exact. The successful build took 204.65 seconds; the preserved failed run took 56.09 seconds. The observed successful peak memory was 1.13 GB. `ACTUAL_BANK_BUILD_RECEIPT.json` records the resource accounting and `POSTBUILD_ARTIFACT_VALIDATION.json` records the independent artifact/prefix checks. All-method finite coverage and all registered legacy-query prediction invariance remain separate adapter gates before TEST opening.
