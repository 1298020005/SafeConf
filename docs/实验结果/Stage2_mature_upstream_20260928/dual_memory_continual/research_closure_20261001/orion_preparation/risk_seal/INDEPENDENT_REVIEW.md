# Independent Source risk sealer review

**PASS.** Reviewed code SHA256:
`8b69e6ce29dc92aa34150b3b08aac62ce8333a10f84ec432a39d0e81ffc64482`.
This review used generated query predictions/history and frozen historical Source
models. It opened no real Orion expression or prediction files and fitted no model.

`INDEPENDENT_REVIEW.json` records ten independent checks. A separate synthetic
execution exercised all six actual persisted Source risk models and the actual
Source Public model; its hash and external runtime location are recorded there.

The learned Public model is trained on historical Source continuous biological
transfer RMSE with the exact nine Source pair features. Its inference explicitly
uses `clip=False`. An independent stub returning RMSE predictions 2 and 4 confirms
that these values remain above 1 and drive the fixed learned prior rule without
CDF/probability clipping. The nine feature names/order match the original Source
helper; synthetic hand expectations cover their values. The six prediction
features match the existing McFaline prediction-feature helper exactly.

Each public risk feature frame contains the fixed six prediction features plus
seven public features. The six predetermined risk candidates are P-only,
Manual-public and Learned-public, each with Ridge and HGB. No query or memory IDs
enter these numeric feature columns; changing query IDs, target Ensembl IDs and
memory experiment IDs leaves all risk scores and history distances unchanged.
Biological target symbols are used for exact Source-history retrieval only.

Uniform, Manual and Learned priors match the original Source prior helper,
including the standard-deviation floor and fixed 50:50 learned/cell-weight blend.
Independent algebra checks confirm that Manual and Learned history distance
`sqrt(prediction_prior_rmse² + prior_uncertainty²)` equals the square root of the
weighted mean squared discrepancy to the Source history effects.

No same-gene Source history produces NaN prior vectors, seven NaN public features
and four unsupported public risk scores. It produces no zero-effect replacement.
The two P-only risk scores remain separate and finite. Extra query truth/error
columns and nonfinite query predictions are rejected by synthetic negative tests.

Code review confirms that the read-only comparison manifest binds the exact risk
code and prospective scientific contract, records the TEST metadata exposure
registry, and authorizes pretruth inference only. Source model/axis/history hashes
and all LM DELTA/query/axis/TRAIN-control/fitted-model/run-manifest/status hashes
are verified before `infer`. The status checks enforce the published fixed PCA,
ridge and seed settings and assert that query truth was not read. The sealer has
no query-truth, target-error or target-CDF input and calls no fitting API.
Historical **Source** CDF metadata is hashed as a Source dependency; no target CDF
is loaded or fitted.

No concrete blocker remains for inference under the exact frozen manifest and
reviewed code hash. These results establish synthetic behavior and code bindings;
they do not constitute a real Orion risk-seal execution or a test-truth opening.
