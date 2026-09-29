# Public History vs same-model Error Memory

DEV/SEEN mechanism experiment; external confirmation remained sealed.

Every method uses the same gene-cluster error-label budget. Error Memory is built only from same-upstream outer-train errors; query labels never enter memory.

## Experiment-lead decision

- **Public History is the stronger information source.** At 50%–100% feedback, it consistently improves Spearman on both upstreams; the paired 95% intervals for the Spearman increment are positive. U20 point gains reach about +0.014 to +0.018 at 75%–100%, although their intervals still cross zero.
- **The registered local Error Memory is not an independent U20 contribution.** `P+ErrorMemory` never has a positive U20 interval. At 10% feedback it significantly harms GAT (dU20 -0.0565, 95% CI [-0.0913, -0.0223]).
- **No complementarity is demonstrated.** Adding Error Memory to Public History has a negative U20 point estimate at every 75%/100% comparison and every interval crosses zero. Small Spearman/AURC gains do not compensate for the primary U20 result.
- This is a feedback-setting mechanism experiment, because every learned risk estimator receives the same sampled error-label budget. It does not replace the separate cold-start table.

Accordingly, SafeConf's paper core remains Universal Prediction Evidence plus legal Historical Experimental Evidence. Error Memory is retained as a negative/conditional deployment analysis, not a claimed main module. A different Error Memory representation would require a new development hypothesis and is not pursued this week.

## Summary

| upstream | budget | method | U20 | Spearman | AURC |
|---|---:|---|---:|---:|---:|
| txpert_exphormer | 10% | P | 0.723968 | 0.700551 | 0.049639 |
| txpert_exphormer | 10% | P+ErrorMemory | 0.717289 | 0.686504 | 0.049900 |
| txpert_exphormer | 10% | P+PublicHistory | 0.724691 | 0.689877 | 0.049751 |
| txpert_exphormer | 10% | P+PublicHistory+ErrorMemory | 0.713030 | 0.674954 | 0.050048 |
| txpert_exphormer | 25% | P | 0.759240 | 0.723710 | 0.049295 |
| txpert_exphormer | 25% | P+ErrorMemory | 0.749359 | 0.716990 | 0.049311 |
| txpert_exphormer | 25% | P+PublicHistory | 0.743935 | 0.727687 | 0.048833 |
| txpert_exphormer | 25% | P+PublicHistory+ErrorMemory | 0.737472 | 0.719979 | 0.048963 |
| txpert_exphormer | 50% | P | 0.759018 | 0.728763 | 0.049146 |
| txpert_exphormer | 50% | P+ErrorMemory | 0.749708 | 0.729997 | 0.048998 |
| txpert_exphormer | 50% | P+PublicHistory | 0.756199 | 0.746386 | 0.048595 |
| txpert_exphormer | 50% | P+PublicHistory+ErrorMemory | 0.758256 | 0.748508 | 0.048482 |
| txpert_exphormer | 75% | P | 0.763114 | 0.733979 | 0.049118 |
| txpert_exphormer | 75% | P+ErrorMemory | 0.764759 | 0.737583 | 0.048836 |
| txpert_exphormer | 75% | P+PublicHistory | 0.784477 | 0.752846 | 0.048491 |
| txpert_exphormer | 75% | P+PublicHistory+ErrorMemory | 0.779652 | 0.756598 | 0.048315 |
| txpert_exphormer | 100% | P | 0.765939 | 0.733376 | 0.049090 |
| txpert_exphormer | 100% | P+ErrorMemory | 0.757740 | 0.733229 | 0.048900 |
| txpert_exphormer | 100% | P+PublicHistory | 0.782265 | 0.751763 | 0.048519 |
| txpert_exphormer | 100% | P+PublicHistory+ErrorMemory | 0.773449 | 0.754567 | 0.048366 |
| txpert_gat | 10% | P | 0.766442 | 0.731592 | 0.049319 |
| txpert_gat | 10% | P+ErrorMemory | 0.708985 | 0.709353 | 0.049553 |
| txpert_gat | 10% | P+PublicHistory | 0.761694 | 0.724088 | 0.049323 |
| txpert_gat | 10% | P+PublicHistory+ErrorMemory | 0.751440 | 0.708624 | 0.049537 |
| txpert_gat | 25% | P | 0.768632 | 0.723335 | 0.049498 |
| txpert_gat | 25% | P+ErrorMemory | 0.749549 | 0.723872 | 0.049302 |
| txpert_gat | 25% | P+PublicHistory | 0.780719 | 0.741791 | 0.048970 |
| txpert_gat | 25% | P+PublicHistory+ErrorMemory | 0.770694 | 0.738871 | 0.048974 |
| txpert_gat | 50% | P | 0.769632 | 0.734468 | 0.049178 |
| txpert_gat | 50% | P+ErrorMemory | 0.761719 | 0.738292 | 0.049150 |
| txpert_gat | 50% | P+PublicHistory | 0.786238 | 0.747744 | 0.048796 |
| txpert_gat | 50% | P+PublicHistory+ErrorMemory | 0.762691 | 0.751787 | 0.048735 |
| txpert_gat | 75% | P | 0.761358 | 0.740392 | 0.049042 |
| txpert_gat | 75% | P+ErrorMemory | 0.763322 | 0.737604 | 0.049162 |
| txpert_gat | 75% | P+PublicHistory | 0.790980 | 0.755387 | 0.048656 |
| txpert_gat | 75% | P+PublicHistory+ErrorMemory | 0.770930 | 0.753711 | 0.048702 |
| txpert_gat | 100% | P | 0.767351 | 0.744474 | 0.048951 |
| txpert_gat | 100% | P+ErrorMemory | 0.769091 | 0.745991 | 0.048934 |
| txpert_gat | 100% | P+PublicHistory | 0.786387 | 0.759314 | 0.048609 |
| txpert_gat | 100% | P+PublicHistory+ErrorMemory | 0.772326 | 0.761483 | 0.048525 |

## Increment intervals

| upstream | budget | comparison | dU20 [95% CI] | dSpearman [95% CI] |
|---|---:|---|---:|---:|
| txpert_exphormer | 10% | PublicHistory_given_P | -0.012876 [-0.054205, +0.027822] | -0.010513 [-0.028496, +0.007066] |
| txpert_exphormer | 10% | ErrorMemory_given_P | -0.015422 [-0.051002, +0.017073] | -0.014061 [-0.021601, -0.006749] |
| txpert_exphormer | 10% | ErrorMemory_given_PublicHistory | -0.009607 [-0.035281, +0.014283] | -0.014977 [-0.022025, -0.008074] |
| txpert_exphormer | 10% | PublicHistory_given_ErrorMemory | -0.007061 [-0.047320, +0.034306] | -0.011429 [-0.029335, +0.006529] |
| txpert_exphormer | 25% | PublicHistory_given_P | -0.014097 [-0.046365, +0.019025] | +0.003987 [-0.010997, +0.018581] |
| txpert_exphormer | 25% | ErrorMemory_given_P | -0.004409 [-0.028855, +0.019476] | -0.006709 [-0.014106, +0.000679] |
| txpert_exphormer | 25% | ErrorMemory_given_PublicHistory | -0.009612 [-0.034994, +0.015300] | -0.007697 [-0.014247, -0.001400] |
| txpert_exphormer | 25% | PublicHistory_given_ErrorMemory | -0.019301 [-0.050990, +0.012328] | +0.002999 [-0.010741, +0.016347] |
| txpert_exphormer | 50% | PublicHistory_given_P | +0.002887 [-0.025841, +0.031870] | +0.017481 [+0.005868, +0.028986] |
| txpert_exphormer | 50% | ErrorMemory_given_P | -0.009157 [-0.029840, +0.011839] | +0.001273 [-0.005425, +0.008016] |
| txpert_exphormer | 50% | ErrorMemory_given_PublicHistory | -0.007257 [-0.030204, +0.014244] | +0.002175 [-0.003787, +0.008342] |
| txpert_exphormer | 50% | PublicHistory_given_ErrorMemory | +0.004788 [-0.023519, +0.034482] | +0.018384 [+0.007170, +0.029601] |
| txpert_exphormer | 75% | PublicHistory_given_P | +0.013989 [-0.014707, +0.043442] | +0.018863 [+0.008466, +0.029438] |
| txpert_exphormer | 75% | ErrorMemory_given_P | +0.000186 [-0.020190, +0.020856] | +0.003592 [-0.003367, +0.011197] |
| txpert_exphormer | 75% | ErrorMemory_given_PublicHistory | -0.002802 [-0.022830, +0.016192] | +0.003734 [-0.002667, +0.010484] |
| txpert_exphormer | 75% | PublicHistory_given_ErrorMemory | +0.011002 [-0.015867, +0.039851] | +0.019005 [+0.008525, +0.029578] |
| txpert_exphormer | 100% | PublicHistory_given_P | +0.014013 [-0.014703, +0.043275] | +0.018247 [+0.006542, +0.029507] |
| txpert_exphormer | 100% | ErrorMemory_given_P | -0.009031 [-0.032299, +0.009836] | -0.000204 [-0.008349, +0.008044] |
| txpert_exphormer | 100% | ErrorMemory_given_PublicHistory | -0.007419 [-0.030503, +0.015142] | +0.002797 [-0.004411, +0.010325] |
| txpert_exphormer | 100% | PublicHistory_given_ErrorMemory | +0.015626 [-0.014013, +0.046065] | +0.021248 [+0.009780, +0.032543] |
| txpert_gat | 10% | PublicHistory_given_P | -0.004327 [-0.033039, +0.024798] | -0.007315 [-0.021290, +0.005912] |
| txpert_gat | 10% | ErrorMemory_given_P | -0.056510 [-0.091284, -0.022262] | -0.022126 [-0.033580, -0.010849] |
| txpert_gat | 10% | ErrorMemory_given_PublicHistory | -0.016237 [-0.051485, +0.017549] | -0.015433 [-0.024530, -0.006689] |
| txpert_gat | 10% | PublicHistory_given_ErrorMemory | +0.035947 [+0.000283, +0.070241] | -0.000622 [-0.014246, +0.012810] |
| txpert_gat | 25% | PublicHistory_given_P | +0.006330 [-0.017971, +0.032297] | +0.018284 [+0.006686, +0.030276] |
| txpert_gat | 25% | ErrorMemory_given_P | -0.016584 [-0.042286, +0.007406] | +0.000447 [-0.007539, +0.008578] |
| txpert_gat | 25% | ErrorMemory_given_PublicHistory | -0.010394 [-0.040518, +0.017982] | -0.002959 [-0.010084, +0.004180] |
| txpert_gat | 25% | PublicHistory_given_ErrorMemory | +0.012520 [-0.024025, +0.047981] | +0.014878 [+0.003229, +0.027182] |
| txpert_gat | 50% | PublicHistory_given_P | +0.015536 [-0.014119, +0.045847] | +0.013137 [+0.001774, +0.024614] |
| txpert_gat | 50% | ErrorMemory_given_P | -0.008645 [-0.028789, +0.012385] | +0.003810 [-0.004946, +0.012816] |
| txpert_gat | 50% | ErrorMemory_given_PublicHistory | -0.013720 [-0.040987, +0.012933] | +0.004053 [-0.003694, +0.012165] |
| txpert_gat | 50% | PublicHistory_given_ErrorMemory | +0.010461 [-0.017703, +0.038860] | +0.013381 [+0.001892, +0.024705] |
| txpert_gat | 75% | PublicHistory_given_P | +0.018144 [-0.008718, +0.045667] | +0.014894 [+0.004979, +0.025357] |
| txpert_gat | 75% | ErrorMemory_given_P | -0.002213 [-0.025062, +0.020482] | -0.002671 [-0.010978, +0.005788] |
| txpert_gat | 75% | ErrorMemory_given_PublicHistory | -0.012705 [-0.038506, +0.011442] | -0.001562 [-0.009341, +0.006723] |
| txpert_gat | 75% | PublicHistory_given_ErrorMemory | +0.007651 [-0.024146, +0.037871] | +0.016003 [+0.005686, +0.026417] |
| txpert_gat | 100% | PublicHistory_given_P | +0.012677 [-0.013149, +0.040048] | +0.014772 [+0.003968, +0.025709] |
| txpert_gat | 100% | ErrorMemory_given_P | -0.004340 [-0.026273, +0.012209] | +0.001537 [-0.006267, +0.009632] |
| txpert_gat | 100% | ErrorMemory_given_PublicHistory | -0.008106 [-0.030438, +0.012530] | +0.002161 [-0.004868, +0.009519] |
| txpert_gat | 100% | PublicHistory_given_ErrorMemory | +0.008911 [-0.020955, +0.038840] | +0.015396 [+0.004832, +0.025847] |
