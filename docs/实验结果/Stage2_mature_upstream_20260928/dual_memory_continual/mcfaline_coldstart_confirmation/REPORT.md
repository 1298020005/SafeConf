# McFaline cold-start external confirmation

The risk predictions and method hashes were committed at `ddbf427` before test
treated expression was aggregated. The test contains 543 tasks, 380
perturbation clusters and three context/treatment strata.

| method | U20 | Spearman | AURC | miss rate |
|---|---:|---:|---:|---:|
| Magnitude | -0.141916 | -0.118309 | 0.023005 | 0.890351 |
| Zero-label Shared HGB | **0.590130** | 0.420956 | 0.020836 | 0.464425 |
| Validation-adapted learned Public HGB | **0.613336** | **0.669527** | **0.019701** | 0.455166 |
| Validation-adapted manual Public HGB | 0.670083 | 0.679207 | 0.019677 | 0.427388 |
| Validation-adapted learned Public Ridge | 0.686724 | 0.667145 | 0.019693 | **0.392788** |

The primary zero-target-error-label comparison improves U20 over magnitude by
`+0.732046`, with a paired perturbation-cluster 95% interval
`[+0.555438,+0.878654]`. All three external strata improve. The preregistered
primary and secondary gates both pass. The learned-vs-manual ordering was not
used to change the frozen method.
