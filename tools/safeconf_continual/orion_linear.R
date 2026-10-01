# Published linear model numerical core, blind-input preparation only.
# Source: const-ae/linear_perturbation_prediction-Paper
# Commit: bfa6eeea2bd145a1af2ec0127a2e808cc38456a9
# Original file SHA256: 49bda2a46b92c2b0b5bf263b957a3890dbdeff2d4837a94e6b0a397e0e38a6f8
# solve_y_axb below is an exact source-text extraction, not a reimplementation.
# This file contains no expression loader, query truth, reporting or training job.

solve_y_axb <- function(Y, A = NULL, B = NULL, A_ridge = 0.01, B_ridge = 0.01){
  stopifnot(is.matrix(Y) || is(Y, "Matrix"))
  stopifnot(is.null(A) || is.matrix(A) || is(A, "Matrix"))
  stopifnot(is.null(B) || is.matrix(B) || is(B, "Matrix"))
  
  center <- rowMeans(Y)
  Y <- Y - center
  
  if(! is.null(A) && ! is.null(B)){
    stopifnot(nrow(Y) == nrow(A))
    stopifnot(ncol(Y) == ncol(B))
    # fit <- lm.fit(kronecker(t(B), A), as.vector(Y))
    tmp <- as.matrix(Matrix::solve(t(A) %*% A + Matrix::Diagonal(ncol(A)) * A_ridge) %*% t(A) %*% Y %*% t(B) %*% Matrix::solve(B %*% t(B) + Matrix::Diagonal(nrow(B)) * B_ridge))
  }else if(is.null(B)){
    fit <- lm.fit(A, Y)
    tmp <- as.matrix(Matrix::solve(t(A) %*% A + Matrix::Diagonal(ncol(A)) * A_ridge) %*% t(A) %*% Y)
  }else if(is.null(A)){
    fit <- lm.fit(t(B), t(Y))
    tmp <- as.matrix(Y %*% t(B) %*% Matrix::solve(B %*% t(B) + Matrix::Diagonal(nrow(B)) * B_ridge))
  }else{
    stop("Either A or B must be non-null")
  }
  tmp[is.na(tmp)] <- 0
  list(K = tmp, center = center)
}


sf_orion_validate_matrix <- function(x, label) {
  if (!is.matrix(x) || !is.numeric(x) || any(!is.finite(x))) {
    stop(label, " must be a finite numeric matrix")
  }
  for (ids in list(rownames(x), colnames(x))) {
    if (is.null(ids) || anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids)) {
      stop(label, " requires unique, nonempty row and column names")
    }
  }
  invisible(TRUE)
}

sf_orion_lm_fit <- function(train_X, baseline, context_id,
                            pca_dim = 10L, ridge_penalty = 0.1, seed = 1L,
                            estimand_id = "mean_cell_log1p_cp4000_v1") {
  # train_X: gene rows x TRAIN-only condition columns, including legal ctrl.
  # Each entry is mean(log1p(4000 * counts_cell / library_size_cell)),
  # with equal weight per legal training cell, never log(normalize(sum cells)).
  # PCA uses processed X, not changes and not control-only expression.
  sf_orion_validate_matrix(train_X, "train_X")
  if (!is.character(context_id) || length(context_id) != 1L ||
      is.na(context_id) || !nzchar(context_id)) stop("one context_id is required")
  if (!identical(estimand_id, "mean_cell_log1p_cp4000_v1")) {
    stop("input processed-X estimand is not the frozen CP4000 cell mean")
  }
  if (length(pca_dim) != 1L || pca_dim != 10L ||
      length(ridge_penalty) != 1L || ridge_penalty != 0.1) {
    stop("published default PCA10/ridge0.1 is fixed for this adapter")
  }
  if (min(dim(train_X)) <= pca_dim) stop("too few training dimensions for PCA10")
  if (!"ctrl" %in% colnames(train_X)) stop("legal training ctrl pseudobulk is required")
  if (!is.numeric(baseline) || length(baseline) != nrow(train_X) ||
      any(!is.finite(baseline)) || is.null(names(baseline)) ||
      anyDuplicated(names(baseline))) stop("finite named context-specific baseline required")
  baseline_matches <- match(rownames(train_X), names(baseline))
  if (anyNA(baseline_matches)) stop("missing baseline readout gene")
  baseline <- baseline[baseline_matches]
  if (max(abs(train_X[, "ctrl"] - baseline)) > 1e-10) {
    stop("training ctrl pseudobulk and legal context baseline do not match")
  }
  clean_condition <- colnames(train_X)
  if (anyNA(match(setdiff(clean_condition, "ctrl"), rownames(train_X)))) {
    stop("training perturbation target is absent from the full readout gene axis")
  }
  if (length(seed) != 1L || !is.finite(seed)) stop("one fixed finite seed is required")
  set.seed(seed)
  # Retain both official calls and their order, including their RNG use.
  pca <- irlba::prcomp_irlba(as.matrix(train_X), n = pca_dim)
  rownames(pca$x) <- rownames(train_X)
  gene_emb <- pca$x
  pca <- irlba::prcomp_irlba(as.matrix(train_X), n = pca_dim)
  rownames(pca$x) <- rownames(train_X)
  pert_emb <- t(pca$x)
  if (!"ctrl" %in% colnames(pert_emb)) {
    pert_emb <- cbind(pert_emb, ctrl = rep(0, nrow(pert_emb)))
  }
  pert_matches <- match(colnames(pert_emb), clean_condition)
  gene_matches <- match(rownames(gene_emb), rownames(train_X))
  if (sum(!is.na(pert_matches)) <= 1L || anyNA(gene_matches)) {
    stop("insufficient or missing training embedding mappings")
  }
  gene_emb_sub <- gene_emb[!is.na(gene_matches), , drop = FALSE]
  pert_emb_training <- pert_emb[, !is.na(pert_matches), drop = FALSE]
  Y <- (train_X - baseline)[na.omit(gene_matches), na.omit(pert_matches), drop = FALSE]
  coefs <- solve_y_axb(Y = Y, A = gene_emb_sub, B = pert_emb_training,
                      A_ridge = ridge_penalty, B_ridge = ridge_penalty)
  if (any(!is.finite(coefs$K)) || any(!is.finite(coefs$center))) {
    stop("nonfinite published LM coefficients")
  }
  structure(list(gene_emb = gene_emb_sub, pert_emb = pert_emb,
                 K = coefs$K, center = coefs$center,
                 baseline = baseline[na.omit(gene_matches)],
                 context_id = context_id, pca_dim = pca_dim,
                 ridge_penalty = ridge_penalty, seed = seed,
                 estimand_id = estimand_id,
                 training_conditions = clean_condition,
                 official_commit = "bfa6eeea2bd145a1af2ec0127a2e808cc38456a9"),
            class = "safeconf_orion_published_lm")
}

sf_orion_lm_predict <- function(model, query_conditions, context_id,
                                output_genes = NULL,
                                output = c("treated", "delta")) {
  # Only immutable query identities enter this API; no query expression argument.
  if (!inherits(model, "safeconf_orion_published_lm")) stop("invalid fitted LM core")
  if (!identical(context_id, model$context_id)) stop("context mismatch; fit contexts separately")
  output <- match.arg(output)
  if (!is.character(query_conditions) || !length(query_conditions) ||
      anyNA(query_conditions) || any(!nzchar(query_conditions)) ||
      anyDuplicated(query_conditions)) stop("unique nonempty query conditions required")
  pert_matches <- match(query_conditions, colnames(model$pert_emb))
  if (anyNA(pert_matches)) stop("query target has no full-readout training-derived embedding")
  genes <- rownames(model$gene_emb)
  if (is.null(output_genes)) output_genes <- genes
  if (!is.character(output_genes) || !length(output_genes) ||
      anyNA(output_genes) || anyDuplicated(output_genes)) stop("valid unique output gene names required")
  gene_matches <- match(output_genes, genes)
  if (anyNA(gene_matches)) stop("output readout gene is absent from the fitted full axis")
  pred <- as.matrix(model$gene_emb[gene_matches, , drop = FALSE] %*% model$K %*%
                    model$pert_emb[, pert_matches, drop = FALSE] + model$center[gene_matches])
  if (output == "treated") pred <- pred + model$baseline[gene_matches]
  dimnames(pred) <- list(output_genes, query_conditions)
  if (any(!is.finite(pred))) stop("nonfinite LM predictions")
  pred
}
