# Public reference unrestricted headroom: DEV diagnostic

Unrestricted simplex and best-single use evaluated biological truth only as unattainable explanatory optima. No fitted model, inference risk score or adopted version changes.

Status: COMPLETE; completed queries 2350/2350; solver failures 0; wall 11.42s.

Gene-balanced reconstruction MSE:

| Domain | Support | Restricted oracle | Unrestricted oracle | Best single | B2 mean over3seeds | B3 mean over3seeds | B2 headroom closed | B3 headroom closed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| McFaline | 0.00091637 | 0.00089391 | 0.00089175 | 0.00141298 | 0.00091169 | 0.00091165 | 19.0% | 19.2% |
| Source | 0.00452369 | 0.00403394 | 0.00396601 | 0.00468058 | 0.00414206 | 0.00414377 | 68.4% | 68.1% |

SLSQP returns numerical optima under fixed tolerance. Per-query Frank-Wolfe dual gaps provide an additional upper bound on the unresolved objective error; failures and ordering violations are retained.
These diagnostic reconstruction limits do not bound risk-ranking Utility@20 and do not identify why risk transfer failed. A remaining reconstruction gap cannot justify outcome-driven model selection or another architecture search.

For equal total gene weights, Source's support-to-unrestricted MSE headroom is 12.328%, and removing the support floor adds 1.684% improvement relative to the restricted oracle. B2/B3 close 68.43%/68.12% of this support-to-oracle headroom; their residual MSE is about 4.25% above the attained unrestricted limit. This is measurable reconstruction space, so missing histories alone cannot explain all observed failures.

McFaline's attained support-to-unrestricted improvement is 2.687%; removing the floor adds only 0.242% relative to the restricted oracle. Its weighted numerical dual-gap bound places the exact optimum between MSE 0.000889219 and 0.000891746, corresponding to approximately 2.69%–2.96% total headroom. B2 closes about 17.2%–19.0% of this small headroom. The networks are near the absolute convex reconstruction limit, while their fraction of the available improvement remains modest. Best-single outcome-informed selection has higher aggregate MSE than support averaging in both domains.

Reproduction (completed version is protected from rerunning):

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONNOUSERSITE=1 /home/yyf/.venvs/txpert-08d82eea/bin/python /home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/tools/scripts/diagnose_safeconf_public_reference_headroom_v1.py --max-seconds 300
```
