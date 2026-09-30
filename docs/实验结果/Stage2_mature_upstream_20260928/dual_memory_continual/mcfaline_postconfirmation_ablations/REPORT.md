# McFaline post-confirmation protocol-specified ablations

These ablations were named in the frozen experiment plan but computed after the
one-shot confirmation. They do not replace or tune the sealed primary method.

| zero-label method | U20 | Spearman | AURC |
|---|---:|---:|---:|
| Universal P | 0.039254 | -0.071457 | 0.022349 |
| Manual Public Memory | **0.638174** | 0.230513 | 0.021388 |
| Learned Public Memory | 0.590130 | **0.420956** | **0.020836** |

Learned Public Memory improves U20 over Universal P by `+0.550876`, cluster CI
`[+0.372000,+0.724736]`. Its U20 difference from manual aggregation is
`-0.048044`, CI `[-0.088364,+0.048446]`; no universal learned-over-manual claim
is made.

Within-stratum Public Memory shuffling reduces mean U20 to `0.001409`
(95% shuffle range `[-0.101949,+0.123557]`, empirical p `0.001996`). A
matched-support content shuffle reduces it to `0.480038` (range
`[0.389257,0.567568]`, empirical p `0.003992`). The result therefore depends on
task-specific historical content beyond stratum identity and support volume.
