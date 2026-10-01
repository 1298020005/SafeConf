args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
repo <- normalizePath(args[[1L]])
root <- normalizePath(args[[2L]])
cli <- file.path(repo, "tools/scripts/run_safeconf_orion_lm_blind.R")
core <- file.path(repo, "tools/safeconf_continual/orion_linear.R")
original <- file.path(dirname(root), "run_linear_pretrained_model.original.R")
source(cli)
source(core)
inputs <- file.path(root, "inputs")
stopifnot(!dir.exists(inputs))
dir.create(inputs)
set.seed(613901L)
genes <- sprintf("ENSG_SYN_%07d", seq_len(38606L))
symbols <- sprintf("S%05d", seq_len(38606L))
symbols[c(10000L, 10001L)] <- "DUP_A"
symbols[c(10002L, 10003L)] <- "DUP_B"
full <- data.frame(gene_id = genes, gene_token_id = seq_along(genes), gene_symbol = symbols)
full_path <- file.path(inputs, "FULL_GENE_AXIS.tsv")
write_tsv(full, full_path)
endpoint <- data.frame(gene_id = genes[1:3285], gene_symbol = symbols[1:3285], source_index = 0:3284)
endpoint_path <- file.path(inputs, "OUTPUT_GENE_AXIS.tsv")
write_tsv(endpoint, endpoint_path)
conditions <- c(genes[1:23], "ctrl")
query_targets <- genes[c(24L, 103L, 5000L, 38606L)]
query_symbols <- symbols[c(24L, 103L, 5000L, 38606L)]
latent_genes <- matrix(rnorm(38606L * 10L), 38606L, 10L)
latent_conditions <- matrix(rnorm(10L * 24L), 10L, 24L)
lambda <- 4 * exp(latent_genes %*% (seq(0.22, 0.025, length.out = 10L) * latent_conditions))
rows <- list()
build_receipt <- function(row) {
  values <- list(schema = "safeconf_orion_lm_blind_inputs_v1", dataset_kind = "SYNTHETIC",
     context_id = row$context_id, estimand_id = "mean_cell_log1p_cp4000_v1",
     allowed_expression_roles = "TRAIN,TRAIN_CONTROL_SOURCE_SCOPE",
     contains_only_authorized_training_rows = "TRUE", query_metadata_only = "TRUE",
     private_expression_values_decoded = "FALSE", actual_upstream_attempt_started = "FALSE")
  for (name in c("train_x", "baseline", "conditions", "queries", "full_gene_axis", "output_gene_axis"))
    values[[paste0(name, "_sha256")]] <- sha256(row[[name]])
  write_values(values, row$receipt)
}
for (context in c("SYNTHETIC_HCT116", "SYNTHETIC_HEK293T")) {
  dir <- file.path(inputs, context)
  dir.create(dir)
  counts <- matrix(rpois(38606L * 24L * 2L, rep(lambda, each = 1L)), nrow = 38606L)
  # Two cells per condition; each retains its complete 38,606-gene denominator.
  if (context == "SYNTHETIC_HEK293T") counts[1:50, ] <- counts[1:50, ] + 40L
  for (j in seq_len(23L)) counts[j, c(j, j + 24L)] <- 0L
  processed <- log1p(sweep(counts, 2L, colSums(counts), "/") * 4000)
  X <- (processed[, 1:24] + processed[, 25:48]) / 2
  dimnames(X) <- list(genes, conditions)
  baseline <- X[, "ctrl"]
  row <- list(context_id = context, train_x = file.path(dir, "TRAIN_X.rds"),
     baseline = file.path(dir, "OWN_CONTEXT_NTC_MEAN.rds"),
     conditions = file.path(dir, "TRAIN_CONDITIONS.tsv"), queries = file.path(dir, "QUERY_METADATA.tsv"),
     full_gene_axis = full_path, output_gene_axis = endpoint_path, receipt = file.path(dir, "INPUT_RECEIPT.tsv"))
  saveRDS(X, row$train_x, compress = FALSE)
  saveRDS(baseline, row$baseline, compress = FALSE)
  write_tsv(data.frame(condition_id = conditions, target_gene_symbol = c(symbols[1:23], "Non-Targeting"),
      context_id = context, role = c(rep("TRAIN", 23L), "TRAIN_CONTROL_SOURCE_SCOPE"), n_cells = 2L), row$conditions)
  write_tsv(data.frame(query_id = paste(context, query_symbols, sep = "|"), target_gene_id = query_targets,
      target_gene_symbol = query_symbols, context_id = context, role = c("VALIDATION", "VALIDATION", "TEST", "TEST")), row$queries)
  write_values(list(input_estimand = "mean_cell_log1p_cp4000_v1", full_library_genes = 38606L,
      n_train_cells = ncol(counts), cell_weights = "equal", query_expression_generated = FALSE,
      max_difference_explicit_mean = max(abs(X - vapply(1:24, function(j)
         rowMeans(log1p(sweep(counts[, c(j, j + 24L), drop = FALSE], 2,
           colSums(counts[, c(j, j + 24L), drop = FALSE]), "/") * 4000)), numeric(38606L))))),
      file.path(dir, "CELL_ESTIMAND_CHECK.tsv"))
  build_receipt(row)
  rows[[context]] <- row
  rm(counts, processed, X, baseline); gc()
}
fit_manifest <- file.path(inputs, "FIT_MANIFEST.tsv")
write_tsv(do.call(rbind, lapply(rows, function(x) as.data.frame(x, stringsAsFactors = FALSE))), fit_manifest)
run_cli <- function(mode, manifest, output, logfile) {
  status <- system2(file.path(R.home("bin"), "Rscript"),
     shQuote(c("--vanilla", cli, "--mode", mode, "--manifest", manifest, "--output-dir", output, "--core", core)),
     stdout = logfile, stderr = logfile, env = c("OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1"))
  status
}
fit_root <- file.path(root, "fit")
stopifnot(run_cli("fit", fit_manifest, fit_root, file.path(root, "FIT.log")) == 0L)

# Evaluate assignments from the pinned original only, never its loader or report.
ast <- parse(original, keep.source = TRUE)
assignment <- function(name) {
  hits <- Filter(function(x) is.call(x) && identical(x[[1L]], as.name("<-")) &&
       is.symbol(x[[2L]]) && identical(x[[2L]], as.name(name)), ast)
  stopifnot(length(hits) == 1L)
  hits[[1L]]
}
official <- new.env(parent = globalenv())
eval(assignment("solve_y_axb"), official)
checks <- list()
record <- function(context, check, diff, tolerance = 1e-10) {
  stopifnot(is.finite(diff), diff <= tolerance)
  checks[[length(checks) + 1L]] <<- data.frame(context_id = context, check = check,
      max_absolute_difference = diff, tolerance = tolerance, pass = TRUE)
}
predict_rows <- list()
for (context in names(rows)) {
  row <- rows[[context]]
  X <- readRDS(row$train_x)
  baseline <- readRDS(row$baseline)
  fitted <- readRDS(file.path(fit_root, context, "MODEL.rds"))
  oracle <- new.env(parent = globalenv())
  oracle$train_data <- X
  oracle$assay <- function(object, assay_name) {stopifnot(assay_name == "X"); object}
  oracle$pa <- list(gene_embedding = "training_data", pert_embedding = "training_data",
       pca_dim = 10L, ridge_penalty = 0.1, seed = 1L)
  oracle$solve_y_axb <- official$solve_y_axb
  set.seed(1L)
  eval(assignment("gene_emb"), oracle)
  eval(assignment("pert_emb"), oracle)
  oracle$pert_emb <- cbind(oracle$pert_emb, ctrl = rep(0, nrow(oracle$pert_emb)))
  record(context, "gene_PCA_full38606_equals_published_AST", max(abs(fitted$gene_emb - oracle$gene_emb)))
  record(context, "pert_PCA_full38606_equals_published_AST", max(abs(fitted$pert_emb - oracle$pert_emb)))
  A <- oracle$gene_emb
  B <- oracle$pert_emb[, match(colnames(X), colnames(oracle$pert_emb)), drop = FALSE]
  Y <- X - baseline
  center <- rowMeans(Y)
  K <- solve(crossprod(A) + diag(0.1, ncol(A)), t(A) %*% (Y - center)) %*%
       t(B) %*% solve(tcrossprod(B) + diag(0.1, nrow(B)))
  coefs <- official$solve_y_axb(Y, A, B, 0.1, 0.1)
  record(context, "ridge_K_equals_independent_base_dense_closed_form", max(abs(fitted$K - K)))
  record(context, "ridge_K_equals_original_solver", max(abs(fitted$K - coefs$K)))
  record(context, "intercept_equals_original_solver", max(abs(fitted$center - coefs$center)))
  q <- read_tsv(row$queries, "query_id")
  dense_pred <- A %*% K %*% oracle$pert_emb[, q$target_gene_id, drop = FALSE] + center + baseline
  saved <- readRDS(file.path(fit_root, context, "PREDICTIONS_TREATED.rds"))
  saved_delta <- readRDS(file.path(fit_root, context, "PREDICTIONS_DELTA.rds"))
  record(context, "pretruth_endpoint_predictions_equal_independent_dense_formula", max(abs(saved - dense_pred[endpoint$gene_id, ])))
  record(context, "delta_equals_treated_minus_own_context_NTC", max(abs(saved_delta - (saved - baseline[endpoint$gene_id]))))
  record(context, "full_readout_embeddings_include_targets_outside_endpoint", as.numeric(!all(q$target_gene_id %in% colnames(fitted$pert_emb))))
  baselines <- readRDS(file.path(fit_root, context, "SIMPLE_BASELINES.rds"))
  effect <- (X - baseline)[endpoint$gene_id, colnames(X) != "ctrl", drop = FALSE]
  independent_mean <- rowMeans(effect)
  independent_zero_RMSE <- mean(apply(effect, 2L, function(x) sqrt(mean(x^2))))
  independent_mean_RMSE <- mean(vapply(seq_len(ncol(effect)), function(j)
     sqrt(mean((effect[, j] - rowMeans(effect[, -j, drop = FALSE]))^2)), 0.0))
  record(context, "simple_baseline_is_unweighted_TRAIN_task_mean_effect", max(abs(baselines$TRAIN_task_unweighted_mean_effect - independent_mean)))
  record(context, "strongest_baseline_selected_on_independent_TRAIN_leave_gene_out_RMSE",
     max(abs(baselines$TRAIN_grouped_OOF_macro_RMSE - c(independent_zero_RMSE, independent_mean_RMSE))))
  stopifnot(baselines$selected == names(baselines$TRAIN_grouped_OOF_macro_RMSE)[which.min(c(independent_zero_RMSE, independent_mean_RMSE))])
  stopifnot(any(!q$target_gene_id %in% endpoint$gene_id), nrow(fitted$gene_emb) == 38606L,
            ncol(fitted$pert_emb) == 38607L, identical(rownames(saved), endpoint$gene_id), identical(colnames(saved), q$query_id))
  predict_rows[[context]] <- data.frame(context_id = context,
       model_bundle = file.path(fit_root, context, "MODEL.rds"), model_sha256 = sha256(file.path(fit_root, context, "MODEL.rds")),
       queries = row$queries, queries_sha256 = sha256(row$queries), output_gene_axis = endpoint_path,
       output_gene_axis_sha256 = sha256(endpoint_path))
  rm(X, baseline, fitted, oracle, A, B, Y, center, K, coefs, dense_pred, saved, saved_delta, baselines, effect); gc()
}
baseline_A <- readRDS(rows[[1L]]$baseline)
baseline_B <- readRDS(rows[[2L]]$baseline)
stopifnot(max(abs(baseline_A - baseline_B)) > 0.01)
record("both_contexts", "NTC_means_kept_context_specific", 0)
predict_manifest <- file.path(inputs, "PREDICT_MANIFEST.tsv")
write_tsv(do.call(rbind, predict_rows), predict_manifest)

# Reload inference succeeds with all TRAIN numeric files unavailable.
numeric_paths <- unlist(lapply(rows, function(row) c(row$train_x, row$baseline)))
stopifnot(all(file.rename(numeric_paths, paste0(numeric_paths, ".unavailable"))))
reload_root <- file.path(root, "reload")
reload_exit <- run_cli("predict", predict_manifest, reload_root, file.path(root, "RELOAD.log"))
stopifnot(all(file.rename(paste0(numeric_paths, ".unavailable"), numeric_paths)), reload_exit == 0L)
for (context in names(rows)) for (kind in c("TREATED", "DELTA")) {
  filename <- paste0("PREDICTIONS_", kind, ".rds")
  a <- readRDS(file.path(fit_root, context, filename))
  b <- readRDS(file.path(reload_root, context, filename))
  stopifnot(identical(a, b), sha256(file.path(fit_root, context, filename)) == sha256(file.path(reload_root, context, filename)))
  record(context, paste0("reload_", tolower(kind), "_prediction_bytes_identical_without_TRAIN_inputs"), 0)
}

# An ambiguous symbol fails before reading a deliberately invalid TRAIN RDS.
bad <- rows[[1L]]
bad_dir <- file.path(root, "ambiguous_target_rejection")
dir.create(bad_dir)
bad$train_x <- file.path(bad_dir, "POISON_TRAIN_X.rds")
writeLines("This is not an RDS and must never be evaluated", bad$train_x)
bad$queries <- file.path(bad_dir, "BAD_QUERY_METADATA.tsv")
q <- read_tsv(rows[[1L]]$queries, "query_id")
q$target_gene_symbol[[1L]] <- "DUP_A"
q$target_gene_id[[1L]] <- genes[[10000L]]
write_tsv(q, bad$queries)
bad$receipt <- file.path(bad_dir, "INPUT_RECEIPT.tsv")
build_receipt(bad)
bad_manifest <- file.path(bad_dir, "MANIFEST.tsv")
write_tsv(as.data.frame(bad), bad_manifest)
bad_log <- file.path(bad_dir, "REJECTION.log")
stopifnot(run_cli("fit", bad_manifest, file.path(bad_dir, "outputs"), bad_log) != 0L,
          any(grepl("absent or ambiguous target symbols", readLines(bad_log), fixed = TRUE)))
record("guard", "ambiguous_target_rejected_before_numeric_pseudobulk_read", 0)
write_tsv(do.call(rbind, checks), file.path(root, "WORKFLOW_COMPARISON.tsv"))
write_values(list(status = "PASS", checks = length(checks), synthetic_contexts = 2L,
   full_gene_count = 38606L, endpoint_gene_count = 3285L, query_count_per_context = 4L,
   max_absolute_difference = max(vapply(checks, function(x) x$max_absolute_difference, 0.0)),
   real_Orion_expression_read = FALSE, real_upstream_attempt_started = FALSE,
   synthetic_models_fitted = 2L, synthetic_fit_cli_calls = 1L, synthetic_reload_cli_calls = 1L,
   query_expression_generated = FALSE, test_truth_read = FALSE, original_loader_executed = FALSE,
   original_full_report_executed = FALSE, pca_dim = 10L, ridge_penalty = 0.1, seed = 1L,
   gpu_hours = 0L, core_sha256 = sha256(core), cli_sha256 = sha256(cli)), file.path(root, "STATUS.tsv"))
cat("FULL_SYNTHETIC_WORKFLOW_PASS", length(checks), "checks; max diff",
    max(vapply(checks, function(x) x$max_absolute_difference, 0.0)), "\n")
