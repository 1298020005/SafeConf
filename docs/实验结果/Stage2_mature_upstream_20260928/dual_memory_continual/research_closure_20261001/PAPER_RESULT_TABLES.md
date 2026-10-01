# Core result tables

| line | method | utility20 | spearman | aurc |
|---|---|---|---|---|
| Exphormer_to_GAT | Learned_WeightedHistoryDistance | 0.605786 | 0.599227 | 0.053083 |
| Exphormer_to_GAT | Learned_hgb | 0.793723 | 0.753558 | 0.048675 |
| Exphormer_to_GAT | Magnitude | 0.768216 | 0.741977 | 0.048831 |
| Exphormer_to_GAT | Manual_WeightedHistoryDistance | 0.589739 | 0.566887 | 0.053591 |
| Exphormer_to_GAT | Manual_hgb | 0.784847 | 0.744611 | 0.048874 |
| Exphormer_to_GAT | Prediction_hgb | 0.762670 | 0.734398 | 0.049132 |
| GAT_to_Exphormer | Learned_WeightedHistoryDistance | 0.626475 | 0.603109 | 0.052629 |
| GAT_to_Exphormer | Learned_hgb | 0.795018 | 0.749545 | 0.048570 |
| GAT_to_Exphormer | Magnitude | 0.758726 | 0.727634 | 0.048927 |
| GAT_to_Exphormer | Manual_WeightedHistoryDistance | 0.565115 | 0.570039 | 0.053188 |
| GAT_to_Exphormer | Manual_hgb | 0.786514 | 0.737704 | 0.048815 |
| GAT_to_Exphormer | Prediction_hgb | 0.764206 | 0.723861 | 0.048996 |
| TxPert_to_McFaline | Learned_WeightedHistoryDistance | 0.614206 | 0.639245 | 0.019812 |
| TxPert_to_McFaline | Learned_hgb | 0.551331 | 0.510874 | 0.020515 |
| TxPert_to_McFaline | Magnitude | -0.141916 | -0.118309 | 0.023005 |
| TxPert_to_McFaline | Manual_WeightedHistoryDistance | 0.604652 | 0.628957 | 0.019845 |
| TxPert_to_McFaline | Manual_hgb | 0.583540 | 0.271321 | 0.021314 |
| TxPert_to_McFaline | Prediction_hgb | 0.042859 | -0.011514 | 0.022245 |

| line | method_a | method_b | delta_utility20 | ci95_lower | ci95_upper |
|---|---|---|---|---|---|
| Exphormer_to_GAT | Manual_hgb | Manual_WeightedHistoryDistance | 0.195109 | 0.151362 | 0.270840 |
| Exphormer_to_GAT | Learned_hgb | Learned_WeightedHistoryDistance | 0.187937 | 0.127291 | 0.247440 |
| GAT_to_Exphormer | Manual_hgb | Manual_WeightedHistoryDistance | 0.221400 | 0.142230 | 0.266120 |
| GAT_to_Exphormer | Learned_hgb | Learned_WeightedHistoryDistance | 0.168543 | 0.117654 | 0.232377 |
| TxPert_to_McFaline | Manual_hgb | Manual_WeightedHistoryDistance | -0.021111 | -0.149550 | 0.031432 |
| TxPert_to_McFaline | Learned_hgb | Learned_WeightedHistoryDistance | -0.062875 | -0.124213 | 0.046060 |
