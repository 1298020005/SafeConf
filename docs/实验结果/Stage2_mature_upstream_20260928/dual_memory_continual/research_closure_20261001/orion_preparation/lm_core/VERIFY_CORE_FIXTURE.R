args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
core_path <- args[[1L]]
out_dir <- args[[2L]]
source(core_path)
original_path <- file.path(out_dir, "run_linear_pretrained_model.original.R")
original_ast <- parse(original_path, keep.source = TRUE)
assignment <- function(name) {
  hits <- Filter(function(x) is.call(x) && identical(x[[1L]], as.name("<-")) &&
                 is.symbol(x[[2L]]) && identical(x[[2L]], as.name(name)), original_ast)
  stopifnot(length(hits) == 1L)
  hits[[1L]]
}
# Evaluate only the original mathematical function assignment. Never source the
# original file, its libraries, loader, baseline retrieval or query reporting.
official_env <- new.env(parent = globalenv())
eval(assignment("solve_y_axb"), envir = official_env)
official_solve <- official_env$solve_y_axb

results <- list()
record <- function(name, difference, tolerance = 1e-10) {
  stopifnot(is.finite(difference), difference <= tolerance)
  results[[length(results) + 1L]] <<- data.frame(
    check = name, max_absolute_difference = difference,
    tolerance = tolerance, pass = TRUE, stringsAsFactors = FALSE)
}
expect_error <- function(name, expression) {
  failed <- inherits(try(force(expression), silent = TRUE), "try-error")
  stopifnot(failed)
  record(name, 0)
}

# Frozen CP4000 estimand on cells with unequal library sizes.
counts_tiny <- matrix(c(1000, 0, 1, 1, 100, 100), nrow = 3L)
processed_tiny <- log1p(sweep(counts_tiny, 2L, colSums(counts_tiny), "/") * 4000)
correct_mean <- rowMeans(processed_tiny)
manual_mean <- vapply(seq_len(nrow(counts_tiny)), function(j)
                      mean(log1p(4000 * counts_tiny[j, ] / colSums(counts_tiny))), 0.0)
record("CP4000_mean_cell_log1p_equals_explicit_estimator", max(abs(correct_mean - manual_mean)))
wrong_logsum <- log1p(4000 * rowSums(counts_tiny) / sum(counts_tiny))
stopifnot(max(abs(correct_mean - wrong_logsum)) > 0.1)
write.csv(data.frame(gene = paste0("tiny", 1:3), mean_cell_logCP4000 = correct_mean,
                     logCP4000_of_summed_counts = wrong_logsum),
          file.path(out_dir, "ESTIMAND_FIXTURE.csv"), row.names = FALSE)

# 64 readout genes, 24 TRAIN-only conditions including ctrl, 5 synthetic cells
# per condition. Query targets remain readout rows but are never train columns.
set.seed(92401L)
genes <- sprintf("G%02d", seq_len(64L))
train_conditions <- c(genes[1:23], "ctrl")
counts <- matrix(rpois(64L * 24L * 5L, lambda = 15), nrow = 64L)
for (j in seq_len(23L)) counts[j, ((j-1L)*5L+1L):(j*5L)] <- 0
processed <- log1p(sweep(counts, 2L, colSums(counts), "/") * 4000)
train_X <- vapply(seq_len(24L), function(j)
                  rowMeans(processed[, ((j-1L)*5L+1L):(j*5L), drop = FALSE]),
                  numeric(64L))
dimnames(train_X) <- list(genes, train_conditions)
baseline <- train_X[, "ctrl"]
query <- c("G24", "G31", "G59", "ctrl")
stopifnot(!any(query[query != "ctrl"] %in% train_conditions))
model <- sf_orion_lm_fit(train_X, baseline, context_id = "synthetic_context_A")

# Independent evaluation of the original PCA assignments and solve assignment,
# with only synthetic TRAIN X available to the original expression environment.
oracle <- new.env(parent = globalenv())
oracle$train_data <- train_X
oracle$assay <- function(object, assay_name) {
  stopifnot(identical(assay_name, "X"))
  object
}
oracle$pa <- list(gene_embedding = "training_data", pert_embedding = "training_data",
                  pca_dim = 10L, ridge_penalty = 0.1, seed = 1L)
oracle$solve_y_axb <- official_solve
set.seed(oracle$pa$seed)
eval(assignment("gene_emb"), envir = oracle)
eval(assignment("pert_emb"), envir = oracle)
if (!"ctrl" %in% colnames(oracle$pert_emb))
  oracle$pert_emb <- cbind(oracle$pert_emb, ctrl = rep(0, nrow(oracle$pert_emb)))
pert_matches <- match(colnames(oracle$pert_emb), colnames(train_X))
gene_matches <- match(rownames(oracle$gene_emb), rownames(train_X))
oracle$gene_emb_sub <- oracle$gene_emb[!is.na(gene_matches), , drop = FALSE]
oracle$pert_emb_training <- oracle$pert_emb[, !is.na(pert_matches), drop = FALSE]
oracle$Y <- (train_X - baseline)[na.omit(gene_matches), na.omit(pert_matches), drop = FALSE]
eval(assignment("coefs"), envir = oracle)
oracle$pert_emb_all <- oracle$pert_emb[, match(query, colnames(oracle$pert_emb)), drop = FALSE]
oracle$baseline <- baseline[na.omit(gene_matches)]
eval(assignment("pred"), envir = oracle)
record("gene_PCA10_matches_original_AST", max(abs(model$gene_emb - oracle$gene_emb_sub)))
record("pert_PCA10_matches_original_AST", max(abs(model$pert_emb - oracle$pert_emb)))
record("bilateral_ridge_K_matches_original_function", max(abs(model$K - oracle$coefs$K)))
record("training_intercept_matches_original_function", max(abs(model$center - oracle$coefs$center)))
pred <- sf_orion_lm_predict(model, query, "synthetic_context_A")
record("inductive_treated_prediction_matches_original_AST", max(abs(pred - oracle$pred)))
delta <- sf_orion_lm_predict(model, query, "synthetic_context_A", output = "delta")
record("delta_equals_official_treated_minus_baseline", max(abs(delta - (oracle$pred - baseline))))
selected <- genes[c(31, 3, 57, 12)]
sub <- sf_orion_lm_predict(model, query, "synthetic_context_A", output_genes = selected)
record("output_axis_subset_preserves_full_axis_fit", max(abs(sub - pred[selected, , drop = FALSE])))
stopifnot(max(abs(pred[, "ctrl"] - baseline)) > 1e-8)
record("ctrl_keeps_official_intercept_not_forced_baseline", 0)

# Second synthetic context has its own TRAIN cells and legal ctrl mean.
counts_B <- counts
counts_B[1:10, ] <- counts_B[1:10, ] + 17L
processed_B <- log1p(sweep(counts_B, 2L, colSums(counts_B), "/") * 4000)
train_X_B <- vapply(seq_len(24L), function(j)
                    rowMeans(processed_B[, ((j-1L)*5L+1L):(j*5L), drop = FALSE]),
                    numeric(64L))
dimnames(train_X_B) <- list(genes, train_conditions)
baseline_B <- train_X_B[, "ctrl"]
model_B <- sf_orion_lm_fit(train_X_B, baseline_B, context_id = "synthetic_context_B")
pred_B <- sf_orion_lm_predict(model_B, query, "synthetic_context_B")
record("second_context_has_its_own_control_baseline", max(abs(model_B$baseline - baseline_B)))
stopifnot(max(abs(model_B$baseline - model$baseline)) > 0.01)
record("two_context_baselines_are_not_pooled", 0)
stopifnot(max(abs(pred_B - pred)) > 0.01)
record("second_context_fit_produces_its_own_predictions", 0)

# Independently compute the closed-form double-sided ridge using base solves.
set.seed(523L)
A <- matrix(rnorm(64L * 7L), 64L)
B <- matrix(rnorm(5L * 24L), 5L)
Y <- matrix(rnorm(64L * 24L), 64L)
center <- rowMeans(Y)
K_ref <- solve(crossprod(A) + diag(0.1, ncol(A)), t(A) %*% (Y - center)) %*%
         t(B) %*% solve(tcrossprod(B) + diag(0.1, nrow(B)))
coefs <- solve_y_axb(Y, A, B, 0.1, 0.1)
record("bilateral_ridge_matches_independent_base_closed_form", max(abs(coefs$K - K_ref)))
record("solver_matches_original_function_random_matrices", max(abs(coefs$K - official_solve(Y,A,B,0.1,0.1)$K)))
expect_error("unknown_query_target_fails_without_zero_fill",
             sf_orion_lm_predict(model, "missing_target", "synthetic_context_A"))
expect_error("mixed_context_prediction_fails",
             sf_orion_lm_predict(model, query, "synthetic_context_B"))
expect_error("missing_output_gene_fails_without_query_drop",
             sf_orion_lm_predict(model, query, "synthetic_context_A", output_genes = "missing_gene"))
expect_error("non_default_PCA_fails",
             sf_orion_lm_fit(train_X, baseline, "synthetic_context_A", pca_dim = 9L))
expect_error("wrong_processed_X_estimand_fails",
             sf_orion_lm_fit(train_X, baseline, "synthetic_context_A", estimand_id = "log_sum"))
expect_error("missing_training_target_mapping_fails",
             sf_orion_lm_fit(train_X[-1L,,drop=FALSE], baseline[-1L], "synthetic_context_A"))
write.csv(do.call(rbind, results), file.path(out_dir, "FIXTURE_COMPARISON.csv"), row.names = FALSE)
write.csv(train_X, file.path(out_dir, "SYNTHETIC_TRAIN_X.csv"))
write.csv(pred, file.path(out_dir, "SYNTHETIC_QUERY_PREDICTIONS.csv"))
writeLines(capture.output(sessionInfo()), file.path(out_dir, "R_SESSION_INFO.txt"))
write.csv(data.frame(package = c("R", "Matrix", "irlba"),
                     version = c(as.character(getRversion()), as.character(packageVersion("Matrix")),
                                 as.character(packageVersion("irlba")))),
          file.path(out_dir, "R_DEPENDENCIES.csv"), row.names = FALSE)
cat("SYNTHETIC_FIXTURE_PASS", length(results), "checks; max diff",
    max(vapply(results, function(x) x$max_absolute_difference, 0.0)), "\n")
