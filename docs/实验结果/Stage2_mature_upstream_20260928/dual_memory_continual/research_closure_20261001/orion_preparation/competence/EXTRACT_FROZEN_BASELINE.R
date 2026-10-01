args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4L)
input <- args[[1L]]; axis_path <- args[[2L]]; output <- args[[3L]]
receipt_path <- args[[4L]]
model <- readRDS(file.path(input, "MODEL.rds"))
provenance <- attr(model, "blind_input_provenance")
receipt_sha <- sub(" .*", "", system2("sha256sum", shQuote(receipt_path), stdout = TRUE))
stopifnot(inherits(model, "safeconf_orion_published_lm"),
          model$pca_dim == 10L, model$ridge_penalty == 0.1, model$seed == 1L,
          provenance$core_sha256 == "6f4f9dbdbe5d147019806b60a4aa6c2fe82d888d1e4d87d71b99807552129b25",
          identical(provenance$receipt_sha256, receipt_sha))
baseline <- readRDS(file.path(input, "SIMPLE_BASELINES.rds"))
stopifnot(identical(provenance$simple_baselines, baseline))
axis <- read.delim(axis_path, stringsAsFactors = FALSE, check.names = FALSE)
selection <- read.delim(file.path(input, "BASELINE_SELECTION.tsv"), stringsAsFactors = FALSE)
stopifnot(nrow(axis) == 3285L, !anyDuplicated(axis$gene_id), nrow(selection) == 2L,
          all(selection$candidate %in% c("zero_effect", "TRAIN_task_unweighted_mean_effect")),
          sum(selection$selected) == 1L,
          identical(as.character(selection$candidate[selection$selected]), baseline$selected))
macro <- baseline$TRAIN_grouped_OOF_macro_RMSE
stopifnot(all(is.finite(macro)), all(macro >= 0),
          baseline$selected == names(macro)[which.min(macro)],
          max(abs(selection$TRAIN_macro_RMSE - unname(macro[selection$candidate]))) <= 1e-12)
vector <- baseline[[baseline$selected]]
stopifnot(is.numeric(vector), identical(names(vector), axis$gene_id), all(is.finite(vector)))
writeBin(unname(vector), paste0(output, ".f64le"), size = 8L, endian = "little")
write.table(data.frame(key = c("selected", "n_genes", "selection_scope", "float_precision"),
    value = c(baseline$selected, length(vector), baseline$selection_scope, "IEEE float64 little-endian")),
    paste0(output, ".tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("FROZEN_TRAIN_BASELINE_EXTRACT_PASS", baseline$selected, "\n")
