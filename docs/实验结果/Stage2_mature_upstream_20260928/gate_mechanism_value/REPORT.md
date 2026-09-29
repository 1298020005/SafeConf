# SafeConf learned-gate mechanism test

DEV/SEEN diagnostic only. Frozen v4 and external confirmation are unchanged.

## Fixed mixtures and learned gate

| upstream | method | U20 | Spearman | AURC |
|---|---|---:|---:|---:|
| e190_gears | V1 | 0.303488 | 0.248873 | 0.226533 |
| e190_gears | fixed_w_0.00 | -0.037673 | -0.110056 | 0.242466 |
| e190_gears | fixed_w_0.25 | -0.029791 | 0.078042 | 0.235840 |
| e190_gears | fixed_w_0.50 | -0.089156 | 0.151233 | 0.228385 |
| e190_gears | fixed_w_0.75 | -0.035796 | 0.220778 | 0.226312 |
| e190_gears | fixed_w_1.00 | 0.027636 | 0.253206 | 0.225750 |
| e190_gears | learned_gate | -0.011379 | 0.202962 | 0.227288 |
| txpert_exphormer | V1 | 0.782265 | 0.751763 | 0.048519 |
| txpert_exphormer | fixed_w_0.00 | 0.759474 | 0.733185 | 0.049095 |
| txpert_exphormer | fixed_w_0.25 | 0.784600 | 0.745768 | 0.048702 |
| txpert_exphormer | fixed_w_0.50 | 0.795312 | 0.752951 | 0.048557 |
| txpert_exphormer | fixed_w_0.75 | 0.780266 | 0.756012 | 0.048493 |
| txpert_exphormer | fixed_w_1.00 | 0.766126 | 0.754321 | 0.048564 |
| txpert_exphormer | learned_gate | 0.791781 | 0.757745 | 0.048471 |
| txpert_gat | V1 | 0.786387 | 0.759314 | 0.048609 |
| txpert_gat | fixed_w_0.00 | 0.773729 | 0.744133 | 0.048970 |
| txpert_gat | fixed_w_0.25 | 0.781905 | 0.754574 | 0.048713 |
| txpert_gat | fixed_w_0.50 | 0.794819 | 0.760674 | 0.048594 |
| txpert_gat | fixed_w_0.75 | 0.781021 | 0.762561 | 0.048567 |
| txpert_gat | fixed_w_1.00 | 0.783340 | 0.760653 | 0.048592 |
| txpert_gat | learned_gate | 0.796294 | 0.764010 | 0.048557 |

## Mechanism decision

- **e190_gears**: best fixed `fixed_w_1.00` U20=0.027636; learned=-0.011379; V1=0.303488; correct branch routing=60.0%; learned-best fixed delta=-0.039015 (95% CI [-0.250848, +0.102147]).
- **txpert_exphormer**: best fixed `fixed_w_0.50` U20=0.795312; learned=0.791781; V1=0.782265; correct branch routing=55.0%; learned-best fixed delta=-0.003530 (95% CI [-0.024524, +0.021109]).
- **txpert_gat**: best fixed `fixed_w_0.50` U20=0.794819; learned=0.796294; V1=0.786387; correct branch routing=60.0%; learned-best fixed delta=+0.001475 (95% CI [-0.015419, +0.021079]).

The gate is considered a core method contribution only if it improves on the best fixed mixture across assets and does not rely on a single family. Otherwise it remains a conditional extension.
