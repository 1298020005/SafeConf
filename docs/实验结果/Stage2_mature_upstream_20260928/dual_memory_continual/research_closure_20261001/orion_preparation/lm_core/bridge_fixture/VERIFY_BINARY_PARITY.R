args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
repo <- normalizePath(args[[1L]]); root <- normalizePath(args[[2L]])
source(file.path(repo, "tools/scripts/run_safeconf_orion_lm_blind.R"))
rows <- read_tsv(file.path(root, "bridge/FIT_MANIFEST.tsv"), "context_id")
checks <- list()
record <- function(context, check, diff, tolerance = 1e-12) {
  stopifnot(is.finite(diff), diff <= tolerance)
  checks[[length(checks) + 1L]] <<- data.frame(context_id = context, check = check,
      max_absolute_difference = diff, tolerance = tolerance, pass = TRUE)
}
for (context in rows$context_id) {
  row <- rows[rows$context_id == context, ]
  binary <- read_matrix(row$train_x)
  rds <- readRDS(file.path(root, "export", paste0(context, "_REFERENCE.rds")))
  stopifnot(identical(dimnames(binary), dimnames(rds)), identical(binary, rds))
  record(context, "little_endian_condition_major_binary_equals_RDS_exactly", max(abs(binary-rds)))
  a <- readRDS(file.path(root, "fit_binary", context, "MODEL.rds"))
  b <- readRDS(file.path(root, "fit_rds", context, "MODEL.rds"))
  for (component in c("gene_emb", "pert_emb", "K", "center", "baseline"))
    record(context, paste0("binary_and_RDS_fit_", component), max(abs(a[[component]]-b[[component]])))
  for (kind in c("TREATED", "DELTA")) {
    filename <- paste0("PREDICTIONS_", kind, ".rds")
    a <- readRDS(file.path(root, "fit_binary", context, filename))
    b <- readRDS(file.path(root, "fit_rds", context, filename))
    c <- readRDS(file.path(root, "reload_binary", context, filename))
    record(context, paste0("binary_and_RDS_pretruth_", kind), max(abs(a-b)))
    stopifnot(identical(a,c),sha256(file.path(root, "fit_binary", context, filename)) ==
                              sha256(file.path(root, "reload_binary", context, filename)))
    record(context,paste0("binary_reload_",kind,"_bytes_identical"),0)
  }
}
write_tsv(do.call(rbind,checks),file.path(root,"BINARY_PARITY_COMPARISON.tsv"))
cat("BINARY_PARITY_PASS",length(checks),"checks; max diff",
    max(vapply(checks,function(x)x$max_absolute_difference,0.0)),"\n")
