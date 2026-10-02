# SafeConf TCBB manuscript package

The submission target is IEEE Transactions on Computational Biology and Bioinformatics. This package contains an English scientific manuscript, supplementary methods/results, the actual compiled PDFs, bibliography, six source-bound figures, numeric tables and runnable source snapshots. Author names and declarations remain explicit author inputs.

| File | Purpose |
|---|---|
| `PAPER.md`, `main.tex`, `build/main.pdf` | Main scientific manuscript and compiled IEEEtran review PDF |
| `SUPPLEMENT.md`, `supplement.tex`, `build/supplement.pdf` | Detailed budgets, fixed controls, exact statistics and audited execution scope |
| `references.bib`, `RELATED_WORK.md` | Primary references/software and precise overlap |
| `CLAIM_EVIDENCE_MATRIX.csv` | Claims, their actual evidence and limits |
| `figures/` | Method diagram, older information/feedback figures and completed five-fold PublicSet figures |
| `evidence/RESULT_MANIFEST.json` | Original source paths, table row counts and SHA-256 |
| `REPRODUCIBILITY.md`, `DATA_ACCESS.md` | Execution entry points, required assets and data links |
| `COVER_LETTER_DRAFT.txt`, `AUTHOR_PLACEHOLDERS.md` | TCBB cover-letter draft and author-confirmed declarations |
| `build/BUILD_STATUS.json`, `build/NUMERIC_CHECKS.json` | Actual compilation outcome and numeric/hash checks |
| `SafeConf_TCBB_20261002.zip`, `SUBMISSION_PACKAGE_STATUS.json` | Checked submission archive and its SHA-256 receipt |

## Rebuild from the repository root

```bash
python docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/paper/refresh_paper_evidence.py
python docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/paper/build_paper.py
python docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/paper/verify_paper.py
python docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/paper/package_submission.py
```

The refresh copies completed results. The build renders both Markdown documents, checks bibliography keys and compiles both PDFs. Verification compares printed numeric tables to the result sources, checks source/copy hashes and PDF freshness, and rejects experimental result placeholders. Packaging requires passed verification. These four commands perform no model fits and no new test-array reads.

Edit the Markdown sources and rebuild the generated mirrors. Older common2840 matrices, frozen Orion3285 results, Frangieh native512 stress results, the 543-task alignment diagnostic and new 542-task McFaline comparison retain their own cohorts and evidence roles. The fixed historical-reader architecture decision is distinct from the completed registered joint-reader result. Source snapshots pin the actual files rather than attributing uncommitted work to an invented release commit.

The archive excludes toolchain binaries, full TeX bundles/caches, model checkpoints and large prediction/biological arrays. `DATA_ACCESS.md` and the original manifests specify those prerequisites. Author completion is required before submission.
