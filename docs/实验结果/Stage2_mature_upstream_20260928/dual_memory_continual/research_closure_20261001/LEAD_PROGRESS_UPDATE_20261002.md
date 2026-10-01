# Actual progress and independent decisions, 2026-10-02

## Learner decision

Fixed TabPFN V2 was run on all20registered DEVfolds. Allfour macro primaryU20 paired95%CIs against the weightedHGB includezero. RetainHGB. Small Spearman changes do not justify selecting a different learner, and noexternal test was used for this decision. See tabpfn_feasibility/.

## Data and prediction decision

Orion is a bounded independent-study replication candidate, with actual HCT116 and HEK293T backgrounds. The frozen40shards total18,137,568,035bytes. It has a usable Source3285 commonreadoutaxis and a published lowcostCPUlinear prediction route. This is still a candidate: no new model has passed validationcompetence yet. Two backgrounds do not become three by subdividing tasks. The study is not whollypristine: knownpreviewtargets stay TRAIN.

The Source3285 models are fitted and hashed. R numericalcore20syntheticchecks pass at4.44e-16. R4.4.1/irlba2.3.5.1 are available; Matrix1.7.6 differs from original1.7.0 and is recorded. This is core/math adaptation, not a complete publishedenvironment reproduction.

## Access correction

The older commonbiology builder selected aggregation rows after reading mixedCSR chunks. Thus zeroTESTaggregation did not mean zeroTESTnumeric materialization. Oldfrozenstores and outputs remain unchanged; they retain this qualification. The futurebuilder now calls allowed_row_spans before CSR data access, with3passingunittests. No claim is made that sharedHDFcompressionblocks contain no privatebytes.

For Orion, ordinaryArrowScanner/fastparquetrow_filter materializes private numeric values before masking. A specialized decoder leaves payloads/dictionaries opaque and converts only metadata-authorized rows;12synthetic page/codec/encoding cases pass. Production integration and independent tests must pass before any realTRAIN/VALIDATION expression run. The test expression contract remains closed.

## Scope

The next actual computation is authorized TRAIN/VALIDATION cellwiseCP4000 biology aggregation, then locked publishedCPUupstream training and competence. This is not another algorithm search. The existing strong baselines and failedSource transfer remain in the matrix. Manuscript/supplement prose is deferred by the user.
