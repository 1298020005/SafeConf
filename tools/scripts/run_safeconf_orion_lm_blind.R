#!/usr/bin/env Rscript
# Blind orchestration only. The published mathematical core is sourced unchanged.
# Fit inputs contain authorized TRAIN pseudobulk, own-context NTC mean and metadata.
# Prediction inputs contain a hashed fitted model and query identities only.

fail <- function(message) stop(message, call. = FALSE)
need <- function(ok, message) if (!isTRUE(ok)) fail(message)
sha256 <- function(path) {
  need(file.exists(path), paste("Missing file:", path))
  line <- system2("sha256sum", shQuote(path), stdout = TRUE)
  need(length(line) == 1L && is.null(attr(line, "status")), "sha256sum failed")
  sub(" .*", "", line)
}
read_tsv <- function(path, columns) {
  x <- read.delim(path, check.names = FALSE, stringsAsFactors = FALSE,
                  colClasses = "character", quote = "", comment.char = "", na.strings = NULL)
  need(all(columns %in% names(x)), paste("Missing columns in", path))
  need(nrow(x) > 0L && !anyNA(x), paste("Empty or missing metadata in", path))
  x
}
unique_ids <- function(x, label) {
  need(length(x) > 0L && all(nzchar(x)) && !anyNA(x) && !anyDuplicated(x),
       paste(label, "must be nonempty and unique"))
}
write_tsv <- function(x, path) {
  connection <- if (grepl("[.]gz$", path)) gzfile(path, "wt") else file(path, "wt")
  on.exit(close(connection))
  write.table(x, connection, sep = "\t", row.names = FALSE, quote = FALSE, na = "")
}
write_values <- function(x, path) {
  write_tsv(data.frame(key = names(x), value = vapply(x, as.character, "")), path)
}
read_matrix <- function(path) {
  if (grepl("[.]rds$", path, ignore.case = TRUE)) return(readRDS(path))
  if (grepl("[.]f64le$", path, ignore.case = TRUE)) {
    sidecar <- read_tsv(paste0(path, ".meta.tsv"), c("key", "value"))
    unique_ids(sidecar$key, "Binary sidecar keys")
    info <- setNames(sidecar$value, sidecar$key)
    for (pair in list(c("schema", "safeconf_orion_float64_le_matrix_v1"),
                     c("dtype", "float64"), c("byte_order", "little"),
                     c("disk_layout", "condition_major_gene_minor")))
      need(pair[[1L]] %in% names(info) && info[[pair[[1L]]]] == pair[[2L]], "Unsupported binary matrix format")
    need(all(c("n_genes", "n_conditions", "binary_bytes", "binary_sha256", "gene_axis_path", "gene_axis_sha256",
               "conditions_path", "conditions_sha256") %in% names(info)), "Binary sidecar fields missing")
    dims <- suppressWarnings(as.numeric(info[c("n_genes", "n_conditions")]))
    need(all(is.finite(dims)) && all(dims > 0) && all(dims == floor(dims)) && prod(dims) <= .Machine$integer.max,
         "Invalid binary matrix shape")
    need(file.info(path)$size == 8 * prod(dims) && as.numeric(info[["binary_bytes"]]) == file.info(path)$size,
         "Binary byte count does not match shape")
    need(sha256(path) == info[["binary_sha256"]], "Binary payload hash differs")
    need(sha256(info[["gene_axis_path"]]) == info[["gene_axis_sha256"]] &&
         sha256(info[["conditions_path"]]) == info[["conditions_sha256"]], "Binary axis hash differs")
    genes <- read_tsv(info[["gene_axis_path"]], "gene_id")$gene_id
    conditions <- read_tsv(info[["conditions_path"]], "condition_id")$condition_id
    unique_ids(genes, "Binary gene IDs"); unique_ids(conditions, "Binary condition IDs")
    need(identical(as.numeric(c(length(genes), length(conditions))), unname(dims)), "Binary axes do not match shape")
    # Python condition-by-gene C-order is R gene-by-condition column order.
    # Set dimensions in place rather than allocating a second numeric matrix.
    values <- readBin(path, what = "double", n = prod(dims), size = 8L, endian = "little")
    need(length(values) == prod(dims), "Truncated binary matrix")
    dim(values) <- as.integer(dims)
    dimnames(values) <- list(genes, conditions)
    return(values)
  }
  x <- read_tsv(path, "gene_id")
  ids <- x$gene_id
  x$gene_id <- NULL
  out <- matrix(as.numeric(as.matrix(x)), nrow = nrow(x),
                dimnames = list(ids, names(x)))
  out
}
read_baseline <- function(path) {
  if (grepl("[.]rds$", path, ignore.case = TRUE)) return(readRDS(path))
  x <- read_tsv(path, c("gene_id", "mean_cell_logCP4000"))
  setNames(as.numeric(x$mean_cell_logCP4000), x$gene_id)
}
read_axis <- function(path, expected_count) {
  x <- read_tsv(path, c("gene_id", "gene_symbol"))
  unique_ids(x$gene_id, "Readout Ensembl IDs")
  need(nrow(x) == expected_count, paste("Expected", expected_count, "gene rows"))
  x
}
check_targets <- function(symbols, ids, full_axis, label) {
  counts <- table(full_axis$gene_symbol)
  need(all(nzchar(symbols)) && all(symbols %in% names(counts)) && all(counts[symbols] == 1L),
       paste(label, "contains absent or ambiguous target symbols"))
  mapped <- full_axis$gene_id[match(symbols, full_axis$gene_symbol)]
  need(identical(unname(ids), unname(mapped)), paste(label, "canonical mapping mismatch"))
}
check_queries <- function(path, context, full_axis) {
  fields <- c("query_id", "target_gene_id", "target_gene_symbol", "context_id", "role")
  header <- strsplit(readLines(path, n = 1L, warn = FALSE), "\t", fixed = TRUE)[[1L]]
  need(setequal(header, fields) && !anyDuplicated(header), "Query file must contain only identity metadata")
  q <- read_tsv(path, c("query_id", "target_gene_id", "target_gene_symbol", "context_id", "role"))
  unique_ids(q$query_id, "Query IDs")
  unique_ids(q$target_gene_id, "Query target IDs")
  need(all(q$context_id == context), "Query context mismatch")
  need(all(q$role %in% c("VALIDATION", "TEST")), "Queries must be held-out identities only")
  check_targets(q$target_gene_symbol, q$target_gene_id, full_axis, "Queries")
  q
}
check_output_axis <- function(path, full_axis) {
  x <- read_axis(path, 3285L)
  unique_ids(x$gene_symbol, "Endpoint symbols")
  check_targets(x$gene_symbol, x$gene_id, full_axis, "Endpoint axis")
  x
}
receipt <- function(path, row) {
  x <- read_tsv(path, c("key", "value"))
  unique_ids(x$key, "Receipt keys")
  x <- setNames(x$value, x$key)
  require_value <- function(key, value) need(key %in% names(x) && x[[key]] == value,
                                            paste("Input receipt mismatch:", key))
  require_value("schema", "safeconf_orion_lm_blind_inputs_v1")
  require_value("context_id", row$context_id)
  require_value("estimand_id", "mean_cell_log1p_cp4000_v1")
  require_value("allowed_expression_roles", "TRAIN,TRAIN_CONTROL_SOURCE_SCOPE")
  require_value("contains_only_authorized_training_rows", "TRUE")
  require_value("query_metadata_only", "TRUE")
  need(x[["dataset_kind"]] %in% c("SYNTHETIC", "AUTHORIZED_TRAIN"), "Unknown dataset kind")
  for (name in c("train_x", "baseline", "conditions", "queries", "full_gene_axis", "output_gene_axis"))
    require_value(paste0(name, "_sha256"), sha256(row[[name]]))
  if (grepl("[.]f64le$", row$train_x, ignore.case = TRUE))
    require_value("train_x_sidecar_sha256", sha256(paste0(row$train_x, ".meta.tsv")))
  if (x[["dataset_kind"]] == "AUTHORIZED_TRAIN") {
    for (name in c("expression_permit", "method_contract", "access_audit")) {
      hash_key <- paste0(name, "_sha256")
      path_key <- paste0(name, "_path")
      need(all(c(hash_key, path_key) %in% names(x)) && grepl("^[a-f0-9]{64}$", x[[hash_key]]),
           paste("Required bound authorization artifact:", name))
      require_value(hash_key, sha256(x[[path_key]]))
    }
    need("upstream_attempt_id" %in% names(x) && nzchar(x[["upstream_attempt_id"]]), "Upstream attempt ID required")
    require_value("private_expression_values_decoded", "FALSE")
  }
  x
}
fit_simple_baselines <- function(train_X, baseline, conditions, endpoint, output_dir) {
  # One column per uniquely mapped TRAIN gene; leave one gene cluster out.
  keep <- conditions$role == "TRAIN"
  effect <- sweep(train_X[endpoint$gene_id, keep, drop = FALSE], 1L, baseline[endpoint$gene_id], "-")
  n <- ncol(effect)
  need(n > 1L, "At least two TRAIN tasks required for baseline selection")
  total <- rowSums(effect)
  zero_rmse <- vapply(seq_len(n), function(j) sqrt(mean(effect[, j]^2)), 0.0)
  mean_rmse <- vapply(seq_len(n), function(j)
     sqrt(mean((effect[, j] - (total - effect[, j]) / (n - 1L))^2)), 0.0)
  macro <- c(zero_effect = mean(zero_rmse), TRAIN_task_unweighted_mean_effect = mean(mean_rmse))
  selected <- names(macro)[which.min(macro)]
  baselines <- list(zero_effect = setNames(rep(0, nrow(endpoint)), endpoint$gene_id),
     TRAIN_task_unweighted_mean_effect = setNames(total / n, endpoint$gene_id),
     selected = selected, TRAIN_grouped_OOF_macro_RMSE = macro,
     selection_scope = "own-context TRAIN leave-gene-cluster-out only; no validation or SafeConf errors")
  saveRDS(baselines, file.path(output_dir, "SIMPLE_BASELINES.rds"), compress = FALSE)
  write_tsv(data.frame(condition_id = conditions$condition_id[keep], zero_effect_RMSE = zero_rmse,
      TRAIN_task_mean_leave_gene_out_RMSE = mean_rmse), file.path(output_dir, "BASELINE_TRAIN_GROUPED_OOF_ERRORS.tsv"))
  write_tsv(data.frame(candidate = names(macro), TRAIN_macro_RMSE = unname(macro),
      selected = names(macro) == selected), file.path(output_dir, "BASELINE_SELECTION.tsv"))
  baselines
}
persist_predictions <- function(model, queries, endpoint, output_dir) {
  targets <- queries$target_gene_id
  treated <- sf_orion_lm_predict(model, targets, model$context_id, endpoint$gene_id, "treated")
  delta <- sf_orion_lm_predict(model, targets, model$context_id, endpoint$gene_id, "delta")
  colnames(treated) <- colnames(delta) <- queries$query_id
  saveRDS(treated, file.path(output_dir, "PREDICTIONS_TREATED.rds"), compress = FALSE)
  saveRDS(delta, file.path(output_dir, "PREDICTIONS_DELTA.rds"), compress = FALSE)
  write_tsv(data.frame(gene_id = rownames(treated), treated, check.names = FALSE),
            file.path(output_dir, "PREDICTIONS_TREATED.tsv.gz"))
  write_tsv(data.frame(gene_id = rownames(delta), delta, check.names = FALSE),
            file.path(output_dir, "PREDICTIONS_DELTA.tsv.gz"))
  write_tsv(queries, file.path(output_dir, "QUERY_IDENTITIES.tsv"))
  write_tsv(endpoint, file.path(output_dir, "OUTPUT_GENE_AXIS.tsv"))
  invisible(list(treated = treated, delta = delta))
}
artifact_manifest <- function(path) {
  paths <- sort(list.files(path, full.names = TRUE))
  paths <- paths[basename(paths) != "ARTIFACT_HASHES.tsv"]
  write_tsv(data.frame(path = basename(paths), bytes = file.info(paths)$size,
                      sha256 = vapply(paths, sha256, "")), file.path(path, "ARTIFACT_HASHES.tsv"))
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (length(args) == 1L && args[[1L]] == "--help") {
    cat("Usage: Rscript run_safeconf_orion_lm_blind.R --mode fit|predict --manifest FILE --output-dir DIR [--core FILE]\n",
        "Fit manifest columns: context_id,train_x,baseline,conditions,queries,full_gene_axis,output_gene_axis,receipt\n",
        "Predict manifest columns: context_id,model_bundle,model_sha256,queries,queries_sha256,output_gene_axis,output_gene_axis_sha256\n",
        "Matrices: numeric RDS, hashed .f64le plus .meta.tsv, or gene_id-first TSV(.gz); baseline: named RDS or gene_id/mean_cell_logCP4000 TSV.\n", sep = "")
    return(invisible(NULL))
  }
  need(length(args) %% 2L == 0L, "Arguments must be --key value pairs")
  opts <- setNames(args[seq(2L, length(args), 2L)], sub("^--", "", args[seq(1L, length(args), 2L)]))
  need(!anyDuplicated(names(opts)) && all(names(opts) %in% c("mode", "manifest", "output-dir", "core")), "Unknown or repeated argument")
  need(all(c("mode", "manifest", "output-dir") %in% names(opts)), "Required: --mode, --manifest, --output-dir")
  need(opts[["mode"]] %in% c("fit", "predict"), "Mode must be fit or predict")
  script_arg <- grep("^--file=", commandArgs(), value = TRUE)
  default_core <- file.path(dirname(dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))), "safeconf_continual/orion_linear.R")
  core <- if ("core" %in% names(opts)) opts[["core"]] else default_core
  core_sha <- sha256(core)
  need(core_sha == "6f4f9dbdbe5d147019806b60a4aa6c2fe82d888d1e4d87d71b99807552129b25", "Published numerical core hash differs")
  source(core)
  out <- opts[["output-dir"]]
  need(!dir.exists(out) || length(list.files(out, all.files = TRUE, no.. = TRUE)) == 0L, "Output directory must be new or empty")
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  fields <- if (opts[["mode"]] == "fit") c("context_id", "train_x", "baseline", "conditions", "queries", "full_gene_axis", "output_gene_axis", "receipt") else
    c("context_id", "model_bundle", "model_sha256", "queries", "queries_sha256", "output_gene_axis", "output_gene_axis_sha256")
  manifest <- read_tsv(opts[["manifest"]], fields)
  unique_ids(manifest$context_id, "Context IDs")
  need(all(grepl("^[A-Za-z0-9_]+$", manifest$context_id)), "Unsafe context directory name")
  summaries <- list()
  for (i in seq_len(nrow(manifest))) {
    row <- as.list(manifest[i, , drop = FALSE])
    target <- file.path(out, row$context_id)
    dir.create(target)
    if (opts[["mode"]] == "fit") {
      r <- receipt(row$receipt, row)
      full <- read_axis(row$full_gene_axis, 38606L)
      need("gene_token_id" %in% names(full), "Full axis needs canonical gene_token_id")
      unique_ids(full$gene_token_id, "Gene tokens")
      endpoint <- check_output_axis(row$output_gene_axis, full)
      conditions <- read_tsv(row$conditions, c("condition_id", "target_gene_symbol", "context_id", "role", "n_cells"))
      unique_ids(conditions$condition_id, "TRAIN condition IDs")
      need(all(conditions$context_id == row$context_id), "TRAIN context mismatch")
      ntc <- conditions$condition_id == "ctrl"
      need(sum(ntc) == 1L && conditions$role[ntc] == "TRAIN_CONTROL_SOURCE_SCOPE" &&
             all(conditions$role[!ntc] == "TRAIN"), "Illegal expression role or missing own-context NTC")
      cells <- suppressWarnings(as.numeric(conditions$n_cells))
      need(all(is.finite(cells)) && all(cells > 0L) && all(cells == floor(cells)), "Invalid TRAIN cell counts")
      check_targets(conditions$target_gene_symbol[!ntc], conditions$condition_id[!ntc], full, "TRAIN conditions")
      q <- check_queries(row$queries, row$context_id, full)
      need(!any(q$target_gene_id %in% conditions$condition_id), "Held-out query target appears in TRAIN")
      # All metadata and receipt checks precede pseudobulk values.
      train_X <- read_matrix(row$train_x)
      baseline <- read_baseline(row$baseline)
      need(identical(rownames(train_X), full$gene_id), "Full readout axis/order mismatch")
      need(identical(colnames(train_X), conditions$condition_id), "TRAIN condition axis/order mismatch")
      need(is.numeric(baseline) && identical(names(baseline), full$gene_id), "NTC baseline axis/order mismatch")
      need(all(is.finite(train_X)) && all(train_X >= 0) && all(is.finite(baseline)) && all(baseline >= 0),
           "Processed TRAIN X and NTC means must be finite nonnegative cell-logCP4000 means")
      model <- sf_orion_lm_fit(train_X, baseline, row$context_id, pca_dim = 10L, ridge_penalty = 0.1, seed = 1L)
      baselines <- fit_simple_baselines(train_X, baseline, conditions, endpoint, target)
      attr(model, "blind_input_provenance") <- list(dataset_kind = r[["dataset_kind"]],
           receipt_sha256 = sha256(row$receipt), input_manifest_sha256 = sha256(opts[["manifest"]]),
           core_sha256 = core_sha, full_gene_axis = full, input_receipt = r,
           simple_baselines = baselines, queries_include_expression = FALSE,
           pca_dim = 10L, ridge_penalty = 0.1, seed = 1L)
      saveRDS(model, file.path(target, "MODEL.rds"), compress = FALSE)
      saveRDS(model$gene_emb, file.path(target, "GENE_EMBEDDING.rds"), compress = FALSE)
      saveRDS(model$pert_emb, file.path(target, "PERTURBATION_EMBEDDING.rds"), compress = FALSE)
      saveRDS(model$K, file.path(target, "RIDGE_COEFFICIENTS.rds"), compress = FALSE)
      write_tsv(data.frame(gene_id = names(model$baseline), training_intercept = model$center,
                           own_context_NTC_mean = unname(model$baseline)), file.path(target, "INTERCEPT_AND_BASELINE.tsv.gz"))
      write_tsv(conditions, file.path(target, "TRAIN_CONDITION_IDENTITIES.tsv"))
      write_tsv(full, file.path(target, "FULL_GENE_AXIS.tsv"))
      predicted <- persist_predictions(model, q, endpoint, target)
      reloaded <- readRDS(file.path(target, "MODEL.rds"))
      reloaded_pred <- sf_orion_lm_predict(reloaded, q$target_gene_id, row$context_id, endpoint$gene_id, "treated")
      need(max(abs(predicted$treated - reloaded_pred)) == 0, "Reloaded model prediction differs")
      dataset_kind <- r[["dataset_kind"]]
      rm(train_X, baseline, conditions, baselines, predicted, reloaded, reloaded_pred)
    } else {
      need(sha256(row$model_bundle) == row$model_sha256, "Model hash mismatch")
      need(sha256(row$queries) == row$queries_sha256, "Query metadata hash mismatch")
      need(sha256(row$output_gene_axis) == row$output_gene_axis_sha256, "Endpoint axis hash mismatch")
      model <- readRDS(row$model_bundle)
      provenance <- attr(model, "blind_input_provenance")
      need(!is.null(provenance) && provenance$core_sha256 == core_sha, "Model provenance missing or core differs")
      need(identical(model$context_id, row$context_id) && model$pca_dim == 10L && model$ridge_penalty == 0.1 && model$seed == 1L, "Model context or locked defaults differ")
      full <- provenance$full_gene_axis
      endpoint <- check_output_axis(row$output_gene_axis, full)
      q <- check_queries(row$queries, row$context_id, full)
      persist_predictions(model, q, endpoint, target)
      dataset_kind <- provenance$dataset_kind
    }
    write_values(list(status = "PASS", mode = opts[["mode"]], context_id = row$context_id,
       dataset_kind = dataset_kind, full_gene_count = nrow(full), endpoint_gene_count = nrow(endpoint),
       query_count = nrow(q), core_sha256 = core_sha, pca_dim = 10, ridge_penalty = 0.1, seed = 1,
       input_estimand = "mean_cell_log1p_cp4000_v1", query_truth_read = FALSE,
       real_upstream_attempt_started = dataset_kind == "AUTHORIZED_TRAIN" && opts[["mode"]] == "fit",
       synthetic_fixture_only = dataset_kind == "SYNTHETIC", gpu_hours = 0), file.path(target, "STATUS.tsv"))
    artifact_manifest(target)
    summaries[[i]] <- data.frame(context_id = row$context_id, mode = opts[["mode"]], dataset_kind = dataset_kind,
       model_sha256 = if (opts[["mode"]] == "fit") sha256(file.path(target, "MODEL.rds")) else row$model_sha256,
       prediction_treated_sha256 = sha256(file.path(target, "PREDICTIONS_TREATED.rds")),
       prediction_delta_sha256 = sha256(file.path(target, "PREDICTIONS_DELTA.rds")))
    rm(model, full, endpoint, q); gc()
  }
  write_tsv(do.call(rbind, summaries), file.path(out, "RUN_MANIFEST.tsv"))
  writeLines(capture.output(sessionInfo()), file.path(out, "R_SESSION_INFO.txt"))
  cat("BLIND_LM_WORKFLOW_PASS", opts[["mode"]], nrow(manifest), "contexts\n")
}

if (sys.nframe() == 0L) main()
