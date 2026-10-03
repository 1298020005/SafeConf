# Current-contract complete system

All rows are the same 212 McFaline holdout tasks and use the holdout truth contract. Source and Target score assets are joined by task_id and rejected if their stored truth differs. `SafeConf_no_target_feedback` uses Source-supervised Ridge with Public fallback. `SafeConf_target_feedback_50` uses the separately trained Target Ridge candidate at 50% feedback. The original PublicRule remains an independent no-error-label baseline.
